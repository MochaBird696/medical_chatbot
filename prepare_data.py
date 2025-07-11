# prepare_data.py
# ----------------
# This script pulls in two Hugging Face datasets (UCSD26 and medical-o1-reasoning-SFT),
# downloads MedQuAD directly via HTTP, scrapes 50 hardcoded CDC topic pages,
# and writes out final_medchat_data.jsonl for training.

import os
import json
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset, concatenate_datasets

# ─── 1) Hugging Face datasets ─────────────────────────────────────────────────


def load_o1_sft():
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    def fn(ex):
        prompt = (ex.get("instruction") or ex.get("prompt") or "").strip()
        answer = (ex.get("response")    or ex.get("completion") or "").strip()
        return {"input": prompt, "target": answer}
    return ds.map(fn, remove_columns=ds.column_names)

# ─── 2) MedQuAD via HTTP ─────────────────────────────────────────────────────

def load_medquad():
    ds = load_dataset("lavita/MedQuAD", split="train")

    def fn(ex):
        # use `or ""` so strip() always sees a string
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer")   or "").strip()
        return {"input": q, "target": a}

    return ds.map(fn, remove_columns=ds.column_names)


# ─── 3) Scrape 50 hardcoded CDC topic pages ─────────────────────────────────
CDC_TOPICS = {
    "Alcohol and Public Health":       "https://www.cdc.gov/alcohol/index.html",
    "Alzheimer's Disease":             "https://www.cdc.gov/aging/alzheimers-disease.htm",
    "Arthritis":                       "https://www.cdc.gov/arthritis/index.htm",
    "Asthma":                          "https://www.cdc.gov/asthma/index.html",
    "Autism Spectrum Disorder":        "https://www.cdc.gov/ncbddd/autism/index.html",
    "Cancer":                          "https://www.cdc.gov/cancer/",
    "Chronic Kidney Disease":          "https://www.cdc.gov/kidneydisease/index.html",
    "COPD":                            "https://www.cdc.gov/copd/index.html",
    "COVID-19":                        "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
    "Diabetes":                        "https://www.cdc.gov/diabetes/index.html",
    "Diet and Nutrition":              "https://www.cdc.gov/nutrition/index.html",
    "Disability and Health":           "https://www.cdc.gov/disabilityandhealth/index.html",
    "Ebola Virus Disease":             "https://www.cdc.gov/vhf/ebola/index.html",
    "Environmental Health":            "https://www.cdc.gov/nceh/index.html",
    "Influenza (Flu)":                 "https://www.cdc.gov/flu/index.htm",
    "Heart Disease":                   "https://www.cdc.gov/heartdisease/index.htm",
    "Hypertension":                    "https://www.cdc.gov/high-blood-pressure/index.html",
    "HIV/AIDS":                        "https://www.cdc.gov/hiv/index.html",
    "HPV":                             "https://www.cdc.gov/hpv/index.html",
    "Immunization and Vaccines":       "https://www.cdc.gov/vaccines/index.html",
    "Injury Prevention":               "https://www.cdc.gov/injury/index.html",
    "Lyme Disease":                    "https://www.cdc.gov/lyme/index.html",
    "Mental Health":                   "https://www.cdc.gov/mentalhealth/index.htm",
    "Motor Vehicle Safety":            "https://www.cdc.gov/motorvehiclesafety/index.html",
    "Obesity":                         "https://www.cdc.gov/obesity/index.html",
    "Oral Health":                     "https://www.cdc.gov/oralhealth/index.html",
    "Pneumonia":                       "https://www.cdc.gov/pneumonia/index.html",
    "Prescription Drug Overdose":      "https://www.cdc.gov/drugoverdose/index.html",
    "STD":                             "https://www.cdc.gov/std/index.htm",
    "Stroke":                          "https://www.cdc.gov/stroke/index.htm",
    "Suicide Prevention":              "https://www.cdc.gov/violenceprevention/suicide/index.html",
    "Tobacco and Smoking":             "https://www.cdc.gov/tobacco/index.htm",
    "Tuberculosis (TB)":               "https://www.cdc.gov/tb/topic/basics/index.html",
    "Vaccine Safety":                  "https://www.cdc.gov/vaccinesafety/index.html",
    "Vision Health":                   "https://www.cdc.gov/visionhealth/index.html",
    "Zika Virus":                      "https://www.cdc.gov/zika/index.html",
    "Monkeypox":                       "https://www.cdc.gov/poxvirus/monkeypox/index.html",
    "Measles":                         "https://www.cdc.gov/measles/index.html",
    "Meningitis":                      "https://www.cdc.gov/meningitis/index.html",
    "Hepatitis":                       "https://www.cdc.gov/hepatitis/index.html",
    "Parkinson's Disease":             "https://www.cdc.gov/aging/resources/quick-facts-parkinsons-disease.html",
    "Lead Poisoning":                  "https://www.cdc.gov/nceh/lead/",
    "Rabies":                          "https://www.cdc.gov/rabies/exposure/index.html",
    "Salmonella":                      "https://www.cdc.gov/salmonella/index.html",
    "Pertussis":                       "https://www.cdc.gov/pertussis/index.html",
    "Polio":                           "https://www.cdc.gov/polio/index.htm",
    "Chronic Liver Disease":           "https://www.cdc.gov/hepatitis/statistics/index.htm",
    "Kidney Health for Life":          "https://www.cdc.gov/kidneydisease/kidney-health-for-life.html"
}

def scrape_cdc():
    entries = []
    for topic, url in CDC_TOPICS.items():
        print(f"Scraping {topic}")
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        body = soup.find('div', class_='col-md-8') or soup.find('main')
        if not body: continue
        for header in body.find_all(['h2','h3']):
            title = header.get_text(strip=True)
            node = header.find_next_sibling()
            texts = []
            while node and node.name not in ('h2','h3'):
                if node.name in ('p','ul'):
                    texts.append(node.get_text(' ', strip=True))
                node = node.find_next_sibling()
            if texts:
                question = f"{title} of {topic}?"
                answer   = '\n'.join(texts)
                entries.append({'input': question, 'target': answer})
    return Dataset.from_list(entries)

# ─── Main: Combine & write JSONL ────────────────────────────────────────────
def main():
    o1   = load_o1_sft()
    mq   = load_medquad()
    cdc  = scrape_cdc()
    combined = concatenate_datasets([o1, mq, cdc]).shuffle(seed=42)
    out_file = 'final_medchat_data.jsonl'
    with open(out_file,'w',encoding='utf-8') as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Wrote {len(combined)} examples ({os.path.getsize(out_file)/(1024*1024):.1f} MB) to {out_file}")

if __name__ == '__main__':
    main()
