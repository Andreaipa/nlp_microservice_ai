#!/usr/bin/env bash
# Despliega el microservicio en AWS ECS Express Mode, o lo actualiza si ya
# existe. Se puede ejecutar tantas veces como haga falta.
#
#   SUPABASE_URL=... SUPABASE_KEY=... deploy/aws/deploy.sh
#
# Qué hace:
#   1. Construye la imagen `deploy` (modelo y catálogo dentro) para ARM64.
#   2. La sube a ECR con una etiqueta única (commit + fecha).
#   3. Crea, si faltan, los roles de IAM, el grupo de logs y el clúster.
#   4. Registra una revisión de la task definition con esa imagen.
#   5. Crea el servicio Express Mode o lo actualiza con la nueva revisión.
#   6. Espera a que la URL HTTPS responda y la muestra.
#
# Guía completa: docs/DESPLIEGUE_AWS.md
set -euo pipefail
cd "$(dirname "$0")/../.."

REGION="${AWS_REGION:-us-east-1}"
SERVICE="${SERVICE_NAME:-ai-cylinder-service}"
REPO="${ECR_REPOSITORY:-ai-cylinder-service}"
CLUSTER="${ECS_CLUSTER:-default}"
CPU="${TASK_CPU:-1024}"        # 1 vCPU: ~1,5 s por foto, ver la guía
MEMORY="${TASK_MEMORY:-2048}"  # el servicio usa ~650 MB
MIN_TASKS="${MIN_TASKS:-1}"
MAX_TASKS="${MAX_TASKS:-2}"    # techo de gasto si alguien satura el servicio
LOG_GROUP="/ecs/${SERVICE}"
export AWS_PAGER="" AWS_REGION="$REGION"

step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for f in models/detector.onnx models/detector.classes.json datasets/catalog.json; do
  [[ -f "$f" ]] || fail "falta $f (no está en git; ver docs/DESPLIEGUE_AWS.md)"
done
if [[ -z "${SUPABASE_URL:-}" || -z "${SUPABASE_KEY:-}" ]]; then
  echo "Aviso: sin SUPABASE_URL y SUPABASE_KEY el servicio sólo conocerá el"
  echo "catálogo estático y no verá los cilindros que se den de alta en la app."
fi

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
DIRTY="$(git diff --quiet HEAD -- app Dockerfile requirements datasets/catalog.json || echo -dirty)"
TAG="$(git rev-parse --short HEAD)${DIRTY}-$(date +%Y%m%d%H%M%S)"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

step "Repositorio ECR ${REPO}"
if ! aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPO" \
    --image-tag-mutability IMMUTABLE \
    --image-scanning-configuration scanOnPush=true >/dev/null
fi
# Sólo se conservan las 10 últimas imágenes: cada una ocupa ~250 MB comprimida.
aws ecr put-lifecycle-policy --repository-name "$REPO" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"conservar 10","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}}]}' \
  >/dev/null

step "Imagen ${TAG}"
docker build --platform linux/arm64 --target deploy -t "$IMAGE" .
# Sin `docker login`: en macOS guarda el token en el llavero y falla si ya
# hay uno anterior creado por otro programa ("The specified item already
# exists in the keychain"). En su lugar, un credential helper temporal entrega
# el token de ECR sólo a este registro, sin guardarlo en ningún sitio del
# usuario. Todo se borra al terminar.
DOCKER_HOST="${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}"
export DOCKER_HOST
PUSH_CONFIG="$(mktemp -d)"
trap 'rm -rf "$PUSH_CONFIG"' EXIT
ECR_TOKEN="$(aws ecr get-login-password)"
cat >"${PUSH_CONFIG}/docker-credential-ecrdeploy" <<HELPER
#!/bin/sh
cat >/dev/null
printf '{"ServerURL":"%s","Username":"AWS","Secret":"%s"}' "${REGISTRY}" "${ECR_TOKEN}"
HELPER
chmod 700 "${PUSH_CONFIG}/docker-credential-ecrdeploy"
printf '{"credHelpers":{"%s":"ecrdeploy"}}' "$REGISTRY" >"${PUSH_CONFIG}/config.json"
PATH="${PUSH_CONFIG}:${PATH}" DOCKER_CONFIG="$PUSH_CONFIG" docker push "$IMAGE"
unset ECR_TOKEN

ensure_role() {  # nombre, servicio de confianza, política administrada por AWS
  if aws iam get-role --role-name "$1" >/dev/null 2>&1; then return; fi
  echo "  creando el rol $1"
  aws iam create-role --role-name "$1" >/dev/null --assume-role-policy-document \
    "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"$2\"},\"Action\":\"sts:AssumeRole\"}]}"
  aws iam attach-role-policy --role-name "$1" --policy-arn "$3"
  CREATED_ROLE=1
}
ensure_linked_role() {  # roles que AWS define para sus servicios; no cuestan nada
  local name="$1" service="$2"
  if aws iam get-role --role-name "$name" >/dev/null 2>&1; then return; fi
  echo "  creando el rol de servicio $name"
  aws iam create-service-linked-role --aws-service-name "$service" >/dev/null
  CREATED_ROLE=1
}
step "Roles de IAM"
CREATED_ROLE=0
# Una cuenta que no ha usado ECS, balanceadores o autoescalado no los tiene, y
# Express Mode falla con "Unable to assume the service linked role".
ensure_linked_role AWSServiceRoleForECS ecs.amazonaws.com
ensure_linked_role AWSServiceRoleForElasticLoadBalancing elasticloadbalancing.amazonaws.com
ensure_linked_role AWSServiceRoleForApplicationAutoScaling_ECSService \
  ecs.application-autoscaling.amazonaws.com
ensure_role ecsTaskExecutionRole ecs-tasks.amazonaws.com \
  arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
ensure_role ecsInfrastructureRoleForExpressServices ecs.amazonaws.com \
  arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
if [[ "$CREATED_ROLE" == 1 ]]; then
  echo "  esperando a que IAM propague los roles nuevos"
  sleep 20
fi

step "Logs ${LOG_GROUP} (30 días)"
aws logs create-log-group --log-group-name "$LOG_GROUP" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 30

step "Clúster ${CLUSTER}"
aws ecs create-cluster --cluster-name "$CLUSTER" >/dev/null

step "Task definition"
TD_FILE="$(mktemp)"
trap 'rm -rf "$PUSH_CONFIG" "$TD_FILE"' EXIT
SERVICE="$SERVICE" CPU="$CPU" MEMORY="$MEMORY" ACCOUNT="$ACCOUNT" IMAGE="$IMAGE" TAG="$TAG" \
REGION="$REGION" SUPABASE_URL="${SUPABASE_URL:-}" SUPABASE_KEY="${SUPABASE_KEY:-}" \
python3 - deploy/aws/task-definition.json "$TD_FILE" <<'PY'
import json, os, string, sys
text = string.Template(open(sys.argv[1]).read()).substitute(os.environ)
task = json.loads(text)
# Sin Supabase no se envían las variables vacías.
main = task["containerDefinitions"][0]
main["environment"] = [e for e in main["environment"] if e["value"]]
json.dump(task, open(sys.argv[2], "w"))
PY
TD_ARN="$(aws ecs register-task-definition --cli-input-json "file://${TD_FILE}" \
  --query taskDefinition.taskDefinitionArn --output text)"
echo "  ${TD_ARN}"

step "Servicio ${SERVICE}"
SERVICE_ARN="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
  --query "services[?status=='ACTIVE'].serviceArn | [0]" --output text 2>/dev/null || true)"
if [[ -z "$SERVICE_ARN" || "$SERVICE_ARN" == "None" ]]; then
  echo "  creando (la primera vez tarda unos minutos: balanceador, certificado, red)"
  SERVICE_ARN="$(aws ecs create-express-gateway-service \
    --service-name "$SERVICE" \
    --cluster "$CLUSTER" \
    --infrastructure-role-arn "arn:aws:iam::${ACCOUNT}:role/ecsInfrastructureRoleForExpressServices" \
    --task-definition-arn "$TD_ARN" \
    --health-check-path /api/v1/health \
    --scaling-target "{\"minTaskCount\":${MIN_TASKS},\"maxTaskCount\":${MAX_TASKS},\"autoScalingMetric\":\"AVERAGE_CPU\",\"autoScalingTargetValue\":70}" \
    --query service.serviceArn --output text)"
else
  echo "  actualizando a la nueva revisión"
  aws ecs update-express-gateway-service --service-arn "$SERVICE_ARN" \
    --task-definition-arn "$TD_ARN" >/dev/null
fi
echo "  ${SERVICE_ARN}"

step "Esperando a que el servicio responda con la versión nueva"
# El nombre se resuelve con un DNS público: si alguien lo consultó antes de
# que existiera, el router puede tener en caché "no existe" durante un rato.
check() {  # ruta
  local ip args=()
  ip="$(dig +short @1.1.1.1 "$ENDPOINT" 2>/dev/null | grep -E '^[0-9.]+$' | head -1 || true)"
  [[ -n "$ip" ]] && args=(--resolve "${ENDPOINT}:443:${ip}")
  curl -fsS --max-time 10 ${args[@]+"${args[@]}"} "https://${ENDPOINT}$1" 2>/dev/null
}
ENDPOINT=""
SERVING=0
for _ in $(seq 1 90); do  # hasta 15 minutos; la primera vez suele tardar ~10
  ENDPOINT="$(aws ecs describe-express-gateway-service --service-arn "$SERVICE_ARN" \
    --query 'service.activeConfigurations[0].ingressPaths[0].endpoint' --output text 2>/dev/null || true)"
  [[ "$ENDPOINT" == "None" ]] && ENDPOINT=""
  if [[ -n "$ENDPOINT" ]] && check /api/v1/version | grep -q "\"build_id\":\"${TAG}\""; then
    SERVING=1
    break
  fi
  sleep 10
done
[[ -n "$ENDPOINT" ]] || fail "el servicio no publicó su URL; revise la consola de ECS"
[[ "$SERVING" == 1 ]] || fail "la versión ${TAG} no respondió en 15 minutos en https://${ENDPOINT}
  revise los eventos:  aws ecs describe-services --cluster ${CLUSTER} --services ${SERVICE}
  y los logs:          aws logs tail ${LOG_GROUP} --since 30m"

step "Esperando a que la versión nueva reciba todo el tráfico"
# Express Mode despliega en canario: la versión nueva recibe al principio
# una fracción pequeña de las peticiones mientras ECS vigila que no dé
# errores, y la anterior sigue atendiendo el resto. Hasta que el despliegue
# termina, un usuario puede recibir respuestas de cualquiera de las dos.
ROLLOUT=""
for _ in $(seq 1 80); do  # hasta 40 minutos
  ROLLOUT="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
    --query "services[0].deployments[?status=='PRIMARY'].rolloutState | [0]" --output text)"
  ACTIVE="$(aws ecs describe-services --cluster "$CLUSTER" --services "$SERVICE" \
    --query "length(services[0].deployments)" --output text)"
  [[ "$ROLLOUT" == "COMPLETED" && "$ACTIVE" == "1" ]] && break
  [[ "$ROLLOUT" == "FAILED" ]] && fail "ECS revirtió el despliegue: la versión ${TAG} dio errores.
  La versión anterior sigue en servicio. Revise: aws logs tail ${LOG_GROUP} --since 30m"
  sleep 30
done
[[ "$ROLLOUT" == "COMPLETED" ]] || echo "Aviso: el despliegue sigue en curso; la versión anterior aún atiende parte del tráfico."

printf '\nListo.\n  URL:      https://%s\n  Imagen:   %s\n  Revisión: %s\n\n' "$ENDPOINT" "$IMAGE" "$TD_ARN"
check /api/v1/ready || true
echo
