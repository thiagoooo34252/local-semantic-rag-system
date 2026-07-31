# Sistema local de recuperación semántica (RAG)

Proyecto completo para la **Pre-entrega 3**: ingiere documentos locales, los divide por tokens, persiste sus embeddings en ChromaDB y responde consultas mediante un pipeline RAG asíncrono construido con LCEL.

El dataset describe la Reserva Natural Laguna Verde, un espacio ficticio creado exclusivamente para esta demostración. La generación real usa OpenAI; toda la suite de pruebas y el harness de ejecución son deterministas, locales y no realizan llamadas de red.

## Inicio rápido

Requisitos: Python 3.12 y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --all-groups
cp .env.example .env
```

Edite `.env` y reemplace únicamente el marcador de `OPENAI_API_KEY`. Los modelos económicos predeterminados son `text-embedding-3-small` para embeddings y `gpt-4o-mini` para chat.

```dotenv
OPENAI_API_KEY=replace_with_your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
RAG_TOP_K=4
RAG_PERSIST_PATH=vectorstore
RAG_COLLECTION_NAME=local_semantic_rag
```

## Ingestión

La configuración predeterminada lee solo los archivos `.md` y `.txt` ubicados directamente en `data/`, en orden alfabético, y persiste la colección en `vectorstore/`.

```bash
uv run python ingest.py
```

Salida esperada en la primera ejecución:

```json
{
  "status": "indexed",
  "source_documents": 4,
  "chunks": 11,
  "fingerprint": "<sha256>"
}
```

Una segunda ejecución sin cambios devuelve `"status": "skipped"`. Si cambian los documentos, la colección, el modelo o la configuración de fragmentación, la colección se reconstruye para que no queden fragmentos obsoletos. Para reconstruir manualmente:

```bash
uv run python ingest.py --force
```

Opciones completas:

```bash
uv run python ingest.py \
  --data-path data \
  --persist-path vectorstore \
  --collection-name local_semantic_rag \
  --force
```

## Consultas

Consulta respaldada por el dataset:

```bash
uv run python rag.py "¿Qué recorrido es accesible y cuánto tiempo dura?"
```

La redacción puede variar, pero la respuesta debe mencionar la Pasarela del Humedal, aproximadamente 45 minutos, y referenciar `02_senderos_y_accesibilidad.txt`.

Consulta trampa, fuera del contexto:

```bash
uv run python rag.py "¿Cuál es la capital de Japón?"
```

Salida exigida:

```json
{
  "answer": "No lo sé",
  "references": []
}
```

## Arquitectura

```text
data/*.md|txt
  -> limpieza conservando párrafos
  -> RecursiveCharacterTextSplitter (500 tokens, solape 50)
  -> IDs y metadatos deterministas
  -> OpenAIEmbeddings
  -> ChromaDB local + manifiesto SHA-256

consulta
  -> RunnablePassthrough.assign
  -> RunnableLambda(retriever.ainvoke, k=4)
  -> transformación de documentos y fuentes permitidas
  -> ChatPromptTemplate
  -> ChatOpenAI asíncrono
  -> PydanticOutputParser[RagResponse]
  -> validación final de referencias recuperadas
```

### Ingestión determinista

- Limpia CRLF, espacios horizontales repetidos y tres o más líneas vacías sin eliminar la separación entre párrafos.
- Cuenta con `cl100k_base` y `disallowed_special=()`, por lo que texto similar a un token especial se trata como datos normales.
- Usa separadores recursivos `['\n\n', '\n', ' ', '']`, `chunk_size=500`, `chunk_overlap=50`, `keep_separator='start'` y limpieza explícita de bordes.
- Vuelve a dividir de forma recursiva cualquier combinación excepcional que supere 500 tokens. Nunca corta bytes ni decodifica porciones inseguras de tokens.
- Genera IDs desde fuente, índice y SHA-256 del contenido. Cada fragmento conserva `source`, `chunk_index`, `content_sha256`, `chunk_id` y `embedding_model`.
- Guarda un manifiesto dentro de `vectorstore/`. Chroma persiste automáticamente; no se llama a una API `.persist()` obsoleta.

### Fundamentación y seguridad

- El contexto se declara como dato no confiable y cualquier instrucción incluida en él debe ignorarse.
- El modelo solo puede responder desde el contexto recuperado.
- Una pregunta no sustentada debe producir exactamente `No lo sé` y `[]`.
- El parser Pydantic rechaza respuestas vacías y fallback con referencias.
- La validación posterior rechaza referencias que no pertenecen a las fuentes recuperadas; no las elimina silenciosamente.
- `.env` y `vectorstore/` están ignorados por Git. Los errores de configuración nunca imprimen la clave.

## Pruebas sin API

Las pruebas fijan `PYTHON_DOTENV_DISABLED=1`, eliminan `OPENAI_API_KEY`, bloquean sockets y usan embeddings y modelos asíncronos falsos. No leen un `.env` del equipo, no descargan modelos y no crean un transporte real de OpenAI.

```bash
PYTHON_DOTENV_DISABLED=1 ANONYMIZED_TELEMETRY=FALSE \
uv run pytest \
  -W error::DeprecationWarning \
  -W error::PendingDeprecationWarning \
  --cov --cov-branch --cov-report=term-missing --cov-fail-under=90
```

El harness offline procesa los cuatro archivos reales, reabre la base persistente, recupera una consulta del dominio y verifica también la ruta trampa:

```bash
PYTHON_DOTENV_DISABLED=1 ANONYMIZED_TELEMETRY=FALSE \
uv run python -m tests.offline_harness
```

## Calidad y reproducibilidad

```bash
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -W error::DeprecationWarning -W error::PendingDeprecationWarning \
  --cov --cov-branch --cov-report=term-missing --cov-fail-under=90
uv build
git diff --check
```

La CI usa Python 3.12, instalación bloqueada, Ruff, Pyright, cobertura de ramas mínima de 90%, construcción de wheel y sdist, búsqueda de patrones de secretos y revisión de espacios de parche.

## Estructura

| Ruta | Responsabilidad |
| --- | --- |
| `ingest.py` | Carga, limpieza, fragmentación, manifiesto y persistencia Chroma. |
| `rag.py` | Recuperación asíncrona, pipeline LCEL y CLI JSON. |
| `models.py` | Contrato Pydantic v2 de respuesta fundamentada. |
| `settings.py` | Valores predeterminados y lectura controlada del entorno. |
| `data/` | Cuatro fuentes coherentes sobre una reserva ficticia. |
| `tests/` | Pruebas unitarias, integración Chroma y harness offline. |

## Lista de comprobación de la rúbrica

- [x] Ingestión directa de `.txt` y `.md` con orden, UTF-8 y errores claros.
- [x] Fragmentación recursiva por tokens en 500/50 y postcondición de 500 tokens.
- [x] ChromaDB local con distancia coseno, IDs, metadatos y manifiesto deterministas.
- [x] Reconstrucción por cambios, `--force`, omisión sin cambios y eliminación de datos obsoletos.
- [x] `async def get_rag_response(query: str)` con recuperación top-k 4.
- [x] Pipeline LCEL genuino desde recuperación hasta parser Pydantic.
- [x] Defensa frente a instrucciones en contexto y fallback exacto.
- [x] Referencias únicas y validadas contra las fuentes recuperadas.
- [x] Pregunta respaldada y pregunta trampa cubiertas por pruebas offline.
- [x] CI reproducible, cobertura de ramas, tipos, formato, build y escaneo de secretos.

## Limitaciones

- La calidad semántica real depende del modelo de embeddings, del modelo de chat y de la cobertura del dataset.
- ChromaDB se ejecuta como almacenamiento local de un solo proyecto; no se implementan concurrencia distribuida, autenticación ni respaldo remoto.
- El fallback está controlado por prompt, parser y validaciones, pero un modelo externo puede devolver una salida inválida; en ese caso la aplicación falla de forma cerrada.
- La suite demuestra el flujo sin red. Para evaluar generación y embeddings reales se necesita una clave propia y se incurre en el costo del proveedor.

## Licencia

MIT. Consulte `LICENSE`.
