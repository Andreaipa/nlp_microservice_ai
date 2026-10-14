# Despliegue en AWS

El servicio se despliega en **Amazon ECS Express Mode**: un contenedor en
Fargate detrás de un balanceador que AWS crea y gestiona, con una URL HTTPS
propia (`https://<servicio>.ecs.us-east-1.on.aws`). Es la opción que AWS
recomienda desde que App Runner dejó de admitir clientes nuevos en abril de 2026.

```
App móvil ──HTTPS──▶ Balanceador (ALB, certificado de AWS)
                          │
                          ▼
                Fargate ARM64 · 1 vCPU · 2 GB
                ai-cylinder-service (imagen `deploy`)
                          │  cada 60 s, sólo lectura
                          ▼
                Supabase · tabla `cilindros`
```

## Por qué así

| Necesidad                                                     | Cómo se cubre                                                       |
| ------------------------------------------------------------- | ------------------------------------------------------------------- |
| La app Android exige HTTPS (Capacitor sirve desde `https://`) | URL y certificado incluidos, sin dominio propio                     |
| Que un fallo no deje a los probadores sin servicio            | ECS reinicia el contenedor si `/api/v1/health` no responde          |
| Desplegar una versión nueva sin riesgo                        | Despliegue progresivo con vuelta atrás automática si da errores 5xx |
| Coste acotado                                                 | Como mucho 2 contenedores, aunque alguien sature el servicio        |

## Requisitos

- AWS CLI configurado con un usuario con permisos de administración.
- Docker (en este equipo, OrbStack).
- Los ficheros que no están en git: `models/detector.onnx` y
  `models/detector.classes.json`.

## Desplegar o actualizar

El mismo comando sirve para la primera vez y para cada actualización:

```bash
SUPABASE_URL=https://<proyecto>.supabase.co SUPABASE_KEY=<clave anon> deploy/aws/deploy.sh
```

La URL y la clave pública son las mismas que usa la app
(`IdentCil-main/src/environments/environment.ts`).

El script (`deploy/aws/deploy.sh`):

1. Construye la imagen `deploy` para ARM64: la de inferencia más el modelo, el
   catálogo y los valores calibrados.
2. La sube a ECR con una etiqueta única (commit y fecha). ECR conserva las 10
   últimas.
3. Crea, si no existen, los roles de IAM, el grupo de logs (30 días) y el
   clúster.
4. Registra una revisión de la task definition
   (`deploy/aws/task-definition.json`).
5. Crea el servicio o lo actualiza con esa revisión.
6. Espera a que la URL responda con la versión nueva y muestra `/ready`.

Variables opcionales: `AWS_REGION` (`us-east-1`), `SERVICE_NAME`
(`ai-cylinder-service`), `TASK_CPU` (`1024`), `TASK_MEMORY` (`2048`),
`MIN_TASKS` (`1`) y `MAX_TASKS` (`2`).

La primera vez, AWS tarda unos 10 minutos en crear el balanceador y la red.

**Las actualizaciones son graduales (canario).** La versión nueva arranca en 3–5
minutos, pero al principio sólo recibe una de cada ocho peticiones mientras ECS
vigila que no dé errores; la anterior atiende el resto. El cambio completo
tarda unos 15–20 minutos más, y hasta entonces un probador puede recibir
respuestas de cualquiera de las dos versiones. El script espera a que termine.
Si la versión nueva da errores, ECS vuelve a la anterior automáticamente.

Si se abre la URL **antes** de que exista, el
router puede recordar durante un buen rato que "no existe" y el navegador no
la encontrará aunque ya funcione. Se soluciona solo, o probando desde otra red
(datos móviles). El script hace sus comprobaciones con un DNS público para no
verse afectado.

Para saber qué versión está desplegada:

```bash
curl https://<url>/api/v1/version
```

El campo `build_id` es la etiqueta de la imagen.

## Tamaño elegido: 1 vCPU y 2 GB

Medido en el despliegue real (Fargate ARM64, us-east-1) con 8 fotos del
dataset enviadas desde la red de desarrollo:

| Medida                               | Valor                   |
| ------------------------------------ | ----------------------- |
| Proceso en el servidor (mediana)     | **4,3 s**               |
| Total visto por el cliente (mediana) | 5,5 s                   |
| Memoria usada                        | ~650 MB de 2 GB         |
| Lecturas                             | idénticas a las locales |

El total incluye subir la foto; las de la prueba son originales de 3–5 MB,
y la app envía como mucho 2048 px, así que en la práctica se sube menos.

Los núcleos Graviton de Fargate son unas tres veces más lentos que los del Mac
de desarrollo, donde la misma foto tarda 1,5 s con 1 núcleo y 1,1 s con 2. Si
4–5 s resulta lento para las pruebas, la palanca es la CPU:

```bash
TASK_CPU=2048 TASK_MEMORY=4096 SUPABASE_URL=... SUPABASE_KEY=... deploy/aws/deploy.sh
```

Se espera que baje a unos 3 s, a cambio de duplicar el coste de Fargate (unos
US$29 más al mes). Reducir `AI_OCR_PREPROCESS_VARIANTS` también acelera, pero
cambia la precisión medida para la tesis.

Si varios probadores envían fotos a la vez, esperan turno: el servicio
procesa una cada vez y mientras tanto sigue respondiendo a `/health`. Si la
CPU media supera el 70%, ECS arranca un segundo contenedor.

## Coste aproximado

Con el servicio encendido las 24 horas, en us-east-1:

| Recurso                             | Al mes          |
| ----------------------------------- | --------------- |
| Fargate ARM64, 1 vCPU y 2 GB        | ~US$29          |
| Balanceador (ALB)                   | ~US$17          |
| Direcciones IPv4 públicas           | ~US$7 – 11      |
| ECR, CloudWatch Logs, transferencia | ~US$1 – 2       |
| **Total**                           | **~US$55 – 60** |

Son estimaciones: consulte la factura real en _Billing → Cost Explorer_ a los
pocos días de desplegar. Conviene crear un aviso de presupuesto en _Billing →
Budgets_.

## Operación

Ver los logs en vivo (JSON, uno por petición):

```bash
aws logs tail /ecs/ai-cylinder-service --follow
```

Estado del servicio:

```bash
aws ecs describe-express-gateway-service --service-arn <arn>
```

**Apagarlo** cuando terminen las pruebas. Borra el servicio y su balanceador, y
con ellos casi todo el coste. La imagen sigue en ECR y `deploy.sh` lo vuelve a
crear, pero con **otra URL**, y habrá que actualizar la app:

```bash
aws ecs delete-express-gateway-service --service-arn <arn>
```

## Qué hay que saber

- **El servicio es público.** Cualquiera con la URL puede enviar fotos. El
  techo de 2 contenedores limita el gasto, pero no hay autenticación. Para
  las pruebas basta; antes de producción convendría una clave de API o
  limitar las peticiones por IP (AWS WAF).
- **La auditoría local no persiste.** Los registros y las fotos que el servicio
  guarda en `datasets/audit` viven en el disco del contenedor y se pierden al
  reiniciarlo o actualizarlo. Lo que sí persiste: los logs en CloudWatch (30
  días, con el `request_id` de cada análisis) y la traza que la app guarda en
  Supabase con cada movimiento (`metodo_identificacion`, `corregido`,
  `confianza`).
- **CORS** admite la app Capacitor (`https://localhost`, `capacitor://localhost`)
  y `ionic serve` (`http://localhost:8100`, `:4200`). Se cambia con
  `AI_CORS_ALLOW_ORIGINS` en el Dockerfile.
- **Supabase:** el servicio lee `cilindros` con la clave pública. Si se
  aplican las políticas RLS propuestas
  (`IdentCil-main/supabase/politicas_rls.sql`), esa lectura sigue permitida.

## Conectar la app

En `IdentCil-main/src/environments/environment.prod.ts`:

```ts
aiServiceUrl: 'https://<url>';
```

y compilar la app en modo producción.
