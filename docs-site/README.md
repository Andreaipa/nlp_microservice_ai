# Sitio de documentación

Documentación del microservicio, construida con [Starlight](https://starlight.astro.build/es/).

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # genera dist/
```

La sección "Referencia de la API" se genera automáticamente desde
`../docs/openapi.json`. Si cambia el contrato del servicio, hay que
regenerar ese fichero:

```bash
python ../training/export_openapi.py --out ../docs/openapi.json
```

Las páginas de `src/content/docs/` provienen de los documentos de `../docs/`.
El script `training/sync_docs.py` las vuelve a sincronizar.
