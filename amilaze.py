import pandas as pd
import requests
import time

# Загрузка исходной таблицы
df = pd.read_csv("R02112.tsv", sep="\t")
print(df.head())

# Функция запроса к Metabolic Atlas
def get_crossrefs(gene_id):
    url = f"https://metabolicatlas.org/api/v2/gotenzymes/genes/{gene_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json().get("crossReferences", {})
            result = {}
            for db_name, entries in data.items():
                if entries:
                    entry = entries[0]
                    db_key = db_name.lower().replace(" ", "_")
                    result[f"{db_key}_id"] = entry["id"]
                    result[f"{db_key}_url"] = entry["url"]
            return result
        else:
            return {}
    except:
        return {}

def get_ncbi_sequence(ncbi_id):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "protein",
        "id": ncbi_id,
        "rettype": "fasta",
        "retmode": "text"
    }
    try:
        r = requests.get(base_url, params=params, timeout=10)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            if lines and lines[0].startswith(">"):
                return "".join(lines[1:])
        return ""
    except Exception as e:
        print(f"Ошибка при запросе NCBI для {ncbi_id}: {e}")
        return ""

# Получение последовательности из UniProt
def get_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?fields=sequence"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("sequence", {}).get("value", "")
        else:
            return ""
    except:
        return ""

print(get_ncbi_sequence('XP_001381892'))
# Сбор всех данных
all_data = []
for i, row in df.iterrows():
    gene_id = row["gene"]
    row_data = row.to_dict()
    print(f"[{i+1}/{len(df)}] Gene: {gene_id} — ищем crossReferences...")
    refs = get_crossrefs(gene_id)
    row_data.update(refs)
    # Получение последовательностей
    if "ncbi_protein_id" in refs:
        row_data["ncbi_sequence"] = get_ncbi_sequence(refs["ncbi_protein_id"])
        time.sleep(1)  # задержка для NCBI
    if "uniprotkb_id" in refs:
        row_data["uniprot_sequence"] = get_uniprot_sequence(refs["uniprotkb_id"])
        time.sleep(1)  # задержка для UniProt

    all_data.append(row_data)
    time.sleep(0.5)  # общая задержка между генами

df_final = pd.DataFrame(all_data)
df_final.to_csv("output_with_sequences.csv")
print("Done")
