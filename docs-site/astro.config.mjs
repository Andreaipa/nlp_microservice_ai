// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightOpenAPI, { openAPISidebarGroups } from 'starlight-openapi';

export default defineConfig({
  site: 'http://localhost:4321',
  integrations: [
    starlight({
      title: 'AI Cylinder Service',
      description:
        'Microservicio de visión artificial para la identificación de ' +
        'cilindros industriales mediante detección, OCR y resolución ' +
        'contra el inventario.',
      defaultLocale: 'root',
      locales: {
        root: { label: 'Español', lang: 'es' },
      },
      plugins: [
        // Genera la referencia de la API a partir del esquema exportado con
        // `python training/export_openapi.py`. Así la documentación no puede
        // desviarse del contrato real del servicio.
        starlightOpenAPI([
          {
            base: 'api',
            label: 'Referencia de la API',
            schema: '../docs/openapi.json',
          },
        ]),
      ],
      customCss: ['./src/styles/custom.css'],
      sidebar: [
        {
          label: 'Empezar aquí',
          items: [
            { label: 'Qué es este sistema', slug: 'index' },
            { label: 'Puesta en marcha', slug: 'guias/puesta-en-marcha' },
            { label: 'Resultados medidos', slug: 'guias/resultados' },
          ],
        },
        {
          label: 'Cómo funciona',
          items: [
            { label: 'Arquitectura', slug: 'sistema/arquitectura' },
            { label: 'Rendimiento y tecnología', slug: 'sistema/rendimiento' },
            { label: 'Umbrales de confianza', slug: 'sistema/umbrales' },
            { label: 'Integración con el backend', slug: 'sistema/integracion' },
          ],
        },
        {
          label: 'Datos y modelo',
          items: [
            { label: 'El dataset', slug: 'datos/dataset' },
            { label: 'Protocolo de captura', slug: 'datos/protocolo-captura' },
            { label: 'Estrategia de anotación', slug: 'datos/anotacion' },
            { label: 'Reconocedor: experimento', slug: 'datos/reconocedor' },
          ],
        },
        {
          label: 'Referencia',
          items: [
            { label: 'Configuración', slug: 'referencia/configuracion' },
            { label: 'Scripts y comandos', slug: 'referencia/scripts' },
            { label: 'Diagnóstico inicial', slug: 'referencia/diagnostico' },
          ],
        },
        ...openAPISidebarGroups,
      ],
    }),
  ],
});
