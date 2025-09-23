import mwxml
import bz2
import json

# Ruta al dump XML de Wikipedia
DUMP_PATH = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/enwiki-latest-pages-articles.xml.bz2"
OUTPUT_PATH = "wikipedia_articles.jsonl"  # un JSON por línea

def extract_articles(dump_path, output_path):
    with bz2.open(dump_path) as f, open(output_path, 'w', encoding='utf-8') as out_file:
        dump = mwxml.Dump.from_file(f)

        for page in dump:
            if not page.namespace == 0:  # Solo artículos (no plantillas, archivos, etc.)
                continue

            if page.redirect:
                continue

            try:
                revision = next(page)  # Primera revisión (más reciente)
                text = revision.text or ""
                text = text.strip()
                if not text:
                    continue

                article = {
                    "title": page.title,
                    "text": text
                }

                out_file.write(json.dumps(article) + "\n")
            except Exception as e:
                print(f"Error en la página {page.title}: {e}")

# Ejecutar la extracción
extract_articles(DUMP_PATH, OUTPUT_PATH)
