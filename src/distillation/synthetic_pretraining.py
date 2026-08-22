"""Resumable million-sample, no-code EN/TR conversational pretraining generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from src.distillation.grounded_pilot import calculate, canonical, model_identity, ollama_json

ROOT = Path(__file__).resolve().parents[2]
ASSISTANT_SYSTEM = "You are a concise, helpful English and Turkish conversational assistant. Do not produce programming code."
GENERATOR_SYSTEM = "Reply as the assistant to the conversation. Use the requested language, be concise and natural, and output only the answer. Never output programming code or markdown code blocks."
NAMES = ["Ada","Mert","Elif","Deniz","Ece","Can","Maya","Noah","Lina","Arda","Zeynep","Emir"]
PLACES = ["Ankara","Istanbul","Izmir","Antalya","London","Berlin","Lisbon","Oslo","Vienna","Prague"]
OBJECTS = ["notebook","backpack","tea set","bicycle","lamp","plant","book","ticket","camera","jacket"]
ACTIVITIES = ["reading","walking","cooking","gardening","painting","journaling","swimming","photography","yoga","learning a language"]
TR_ACTIVITIES = ["kitap okumak","yürüyüş yapmak","yemek pişirmek","bahçeyle ilgilenmek","resim yapmak","günlük tutmak","yüzmek","fotoğraf çekmek","yoga yapmak","dil öğrenmek"]
TONES = ["friendly","calm","encouraging","direct","warm","practical"]
TOPICS = ["habits","friendship","motivation","travel planning","sleep routine","time management","home organization","communication","learning","well-being"]
TR_TOPICS = ["alışkanlıklar","arkadaşlık","motivasyon","seyahat planı","uyku düzeni","zaman yönetimi","ev düzeni","iletişim","öğrenme","iyi yaşam"]
TR_MARKERS = {"bir","ve","için","nasıl","nedir","var","olarak","bana","kısa","lütfen","olur","ile","bu","çok","daha"}


def hash_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def language_for(index: int) -> str:
    slot = index % 20
    return "en" if slot < 9 else "tr" if slot < 18 else "bilingual"


def category_for(index: int) -> str:
    slot = index % 100
    if slot < 30: return "conversation"
    if slot < 50: return "contextual_qa"
    if slot < 65: return "transformation"
    if slot < 80: return "reasoning"
    if slot < 90: return "general_qa"
    return "multi_turn"


def make_spec(index: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + index * 1_000_003); lang = language_for(index); category = category_for(index)
    name, other = rng.sample(NAMES, 2); place = rng.choice(PLACES); obj = rng.choice(OBJECTS); number = 10_000 + index
    required: list[str] = []
    if category == "conversation":
        if lang == "tr": user = f"{rng.choice(TR_TOPICS)} konusunda zorlanan birine iki kısa ve uygulanabilir öneri ver. Durum numarası {number}."
        elif lang == "bilingual": user = f"Give one short suggestion about {rng.choice(TOPICS)} in English, then repeat it naturally in Turkish. Scenario {number}."
        else: user = f"Give two short, practical suggestions to someone struggling with {rng.choice(TOPICS)}. Scenario {number}."
    elif category == "contextual_qa":
        count = 2 + rng.randrange(8); color = rng.choice(["red","blue","green","yellow","white"]); tr_color={"red":"kırmızı","blue":"mavi","green":"yeşil","yellow":"sarı","white":"beyaz"}[color]
        if lang == "tr": user=f"{number} numaralı kayıt: {name}, {place} kentindeki {tr_color} {obj} için {count} adet ayırdı. {name} kaç adet ayırdı?"; required=[str(count)]
        elif lang == "bilingual": user=f"Record {number}: {name} reserved {count} {color} items in {place}. Answer the quantity first in English and then in Turkish."; required=[str(count)]
        else: user=f"Record {number}: {name} reserved {count} {color} {obj}s in {place}. How many did {name} reserve?"; required=[str(count)]
    elif category == "transformation":
        activity = rng.choice(TR_ACTIVITIES if lang == "tr" else ACTIVITIES); day=1+rng.randrange(28)
        source = f"{name}, {activity} için ayın {day}. gününde {place} kentinde {other} ile buluşacak." if lang=="tr" else f"On day {day}, {name} will meet {other} in {place} for {activity}."
        if lang == "bilingual": source=f"On day {day}, {name} will meet {other} in {place}."; user=f"Summarize this in one short English sentence and one Turkish sentence: {source}"
        else: user=("Bu cümleyi anlamını değiştirmeden daha kısa yaz: " if lang=="tr" else "Rewrite this more concisely without changing its meaning: ")+source
        required=[name,place]
    elif category == "reasoning":
        a=3+(index*17)%997; b=2+(index*31)%211; c=2+(index*7)%8; result=calculate(f"({a}+{b})*{c}")
        if lang=="tr": user=f"Her kutuda {a} kalem ve {b} silgi var. {c} kutuda toplam kaç eşya vardır? Yalnızca kısa cevap ver."; required=[str(result)]
        elif lang=="bilingual": user=f"Each box has {a} pencils and {b} erasers. There are {c} boxes. Give the total briefly in English and Turkish."; required=[str(result)]
        else: user=f"Each box has {a} pencils and {b} erasers. How many items are in {c} boxes? Answer briefly."; required=[str(result)]
    elif category == "general_qa":
        topic=rng.choice(TR_TOPICS if lang=="tr" else TOPICS)
        if lang=="tr": user=f"{topic} hakkında yeni başlayan birinin sorabileceği basit bir soruyu kısa ve yararlı biçimde yanıtla. Örnek {number}."
        elif lang=="bilingual": user=f"Answer a beginner's everyday question about {rng.choice(TOPICS)} briefly in English and Turkish. Example {number}."
        else: user=f"Briefly answer a simple beginner's everyday question about {topic}. Example {number}."
    else:
        earlier = f"Son zamanlarda {rng.choice(TR_ACTIVITIES)} istiyorum." if lang=="tr" else f"I want to start {rng.choice(ACTIVITIES)} soon."
        reply = "Bunu küçük bir adımla başlatabiliriz." if lang=="tr" else "We can start with one small step."
        if lang=="bilingual": earlier=f"I want to start {rng.choice(ACTIVITIES)} soon."; reply="Let's make it manageable."; user="Suggest the first step in English, then Turkish."
        else: user="İlk adımım ne olsun?" if lang=="tr" else "What should my first step be?"
        return {"id":f"syn-{index:09d}","group":f"syn-{index:09d}","category":category,"language":lang,"messages":[{"role":"system","content":ASSISTANT_SYSTEM},{"role":"user","content":earlier},{"role":"assistant","content":reply},{"role":"user","content":user}],"required":required}
    return {"id":f"syn-{index:09d}","group":f"syn-{index:09d}","category":category,"language":lang,"messages":[{"role":"system","content":ASSISTANT_SYSTEM},{"role":"user","content":user}],"required":required}


def valid(text: str, spec: dict[str, Any], config: dict[str, Any]) -> tuple[bool,str]:
    words=text.split()
    if len(words)<config["min_words"]: return False,"too_short"
    if len(words)>config["max_words"]: return False,"too_long"
    low=canonical(text); numeric_low=re.sub(r"(?<=\d)[ ,.](?=\d)","",text)
    if any((canonical(value) not in low and str(value) not in numeric_low) for value in spec["required"]): return False,"required_value_missing"
    if "```" in text or re.search(r"\b(import|def|class|function|javascript|python|html|css)\b",text,re.I): return False,"code_like"
    if re.search(r"\b(as an ai|bir yapay zeka olarak)\b",low): return False,"model_disclaimer"
    tokens=low.split()
    if tokens and Counter(tokens).most_common(1)[0][1]/len(tokens)>0.42: return False,"token_repetition"
    tr_score=sum(token in TR_MARKERS for token in tokens)+sum(char in text for char in "çğıöşüÇĞİÖŞÜ")
    if spec["language"]=="tr" and tr_score<1: return False,"wrong_language"
    if spec["language"]=="bilingual" and tr_score<1: return False,"missing_turkish"
    return True,"accepted"


def infer(config: dict[str,Any], spec: dict[str,Any], attempt: int) -> tuple[str,dict[str,Any]]:
    language={"en":"English","tr":"Turkish","bilingual":"English followed by Turkish"}[spec["language"]]
    grounding=(" You must preserve these already-verified values or names in the answer: "+", ".join(spec["required"])+". Do not recompute or alter them.") if spec["required"] else ""
    messages=[{"role":"system","content":GENERATOR_SYSTEM+f" Required output language: {language}."+grounding},*spec["messages"][1:]]
    payload={"model":config["model"],"stream":False,"think":False,"messages":messages,"options":{"seed":config["seed"]+int(spec["id"].split("-")[-1])+attempt,"temperature":config["temperature"] if attempt==0 else 0,"num_ctx":config["num_ctx"],"num_predict":config["num_predict"]},"keep_alive":-1}
    response=ollama_json(config["base_url"],"/api/chat",payload)
    metrics={key:response.get(key) for key in ("created_at","done_reason","total_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration")}
    return (response.get("message",{}).get("content") or "").strip(),metrics


class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int):
        self.output_dir=output_dir; self.shards=output_dir/"shards"; self.shards.mkdir(parents=True,exist_ok=True); self.shard_size=shard_size
        self.accepted=0; self.hashes:set[str]=set(); self.rows:list[dict[str,Any]]=[]; self.max_index=-1
        for path in sorted(self.shards.glob("shard-*.jsonl")):
            for line in path.read_text().splitlines():
                row=json.loads(line); self.accepted+=1; self.hashes.add(row["content_sha256"]); self.max_index=max(self.max_index,int(row["id"].split("-")[-1]))
        partial=self.shards/"current.partial.jsonl"
        if partial.exists():
            for line in partial.read_text().splitlines():
                if line.strip(): row=json.loads(line); self.rows.append(row); self.accepted+=1; self.hashes.add(row["content_sha256"]); self.max_index=max(self.max_index,int(row["id"].split("-")[-1]))
    def add(self,row:dict[str,Any])->bool:
        if row["content_sha256"] in self.hashes:return False
        self.hashes.add(row["content_sha256"]);self.rows.append(row);self.accepted+=1
        if len(self.rows)>=self.shard_size:self.commit()
        elif len(self.rows)%25==0:self.sync_partial()
        return True
    def sync_partial(self):
        path=self.shards/"current.partial.jsonl"; path.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in self.rows))
    def commit(self):
        if not self.rows:return
        index=len(list(self.shards.glob("shard-*.jsonl"))); temp=self.shards/"current.partial.jsonl"; self.sync_partial(); temp.replace(self.shards/f"shard-{index:06d}.jsonl"); self.rows=[]


def write_progress(output_dir:Path,config:dict[str,Any],writer:ShardWriter,attempted:int,rejections:Counter,start:float,identity:dict[str,Any]):
    progress={"schema_version":1,"target_accepted":config["target_accepted"],"accepted":writer.accepted,"attempted_specs":attempted,"completion":writer.accepted/config["target_accepted"],"elapsed_seconds":round(time.monotonic()-start,1),"accepted_per_second":round(writer.accepted/max(time.monotonic()-start,0.001),3),"rejections":dict(rejections),"model":identity,"updated_at_unix":time.time()}
    temp=output_dir/"progress.tmp";temp.write_text(json.dumps(progress,indent=2,sort_keys=True)+"\n");temp.replace(output_dir/"progress.json")


def run(config_path:Path,max_new:int|None=None,output_override:Path|None=None):
    config=json.loads(config_path.read_text()); output_dir=output_override or ROOT/config["output_dir"]; output_dir.mkdir(parents=True,exist_ok=True)
    identity=model_identity(config["base_url"],config["model"]); writer=ShardWriter(output_dir,config["shard_size"]); initial=writer.accepted; attempted=writer.accepted; rejects=Counter(); start=time.monotonic(); reject_path=output_dir/"rejections.jsonl"
    last_reported=writer.accepted
    with reject_path.open("a") as reject_file:
        index=writer.max_index+1
        while writer.accepted<config["target_accepted"] and (max_new is None or writer.accepted-initial<max_new):
            spec=make_spec(index,config["seed"]); accepted=False
            for attempt in range(config["max_attempts"]):
                try:text,metrics=infer(config,spec,attempt)
                except Exception as exc:
                    reason=f"request_error:{type(exc).__name__}";rejects[reason]+=1;time.sleep(min(2**attempt,8));continue
                ok,reason=valid(text,spec,config)
                if ok:
                    messages=[*spec["messages"],{"role":"assistant","content":text}]; digest=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                    row={"id":spec["id"],"group":spec["group"],"split":"validation" if hash_fraction(spec["group"])<config["validation_fraction"] else "train","category":spec["category"],"language":spec["language"],"messages":messages,"content_sha256":digest,"generation":{"seed":config["seed"]+index+attempt,"attempt":attempt,"metrics":metrics}}
                    if writer.add(row):accepted=True;break
                    reason="exact_duplicate"
                rejects[reason]+=1;reject_file.write(json.dumps({"id":spec["id"],"attempt":attempt,"reason":reason,"output":text},ensure_ascii=False)+"\n");reject_file.flush()
            attempted+=1
            index+=1
            if writer.accepted-last_reported>=100:
                last_reported=writer.accepted;write_progress(output_dir,config,writer,attempted,rejects,start,identity);print(f"accepted={writer.accepted:,}/{config['target_accepted']:,} rate={(writer.accepted-initial)/max(time.monotonic()-start,.001):.2f}/s",flush=True)
    writer.sync_partial();write_progress(output_dir,config,writer,attempted,rejects,start,identity)
    manifest={"schema_version":1,"config":config,"config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),"model":identity,"accepted":writer.accepted,"shard_size":config["shard_size"],"complete":writer.accepted>=config["target_accepted"]}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,default=ROOT/"configs/synthetic_pretraining_1m.json");parser.add_argument("--max-new",type=int);parser.add_argument("--output-dir",type=Path);args=parser.parse_args();run(args.config,args.max_new,args.output_dir)


if __name__=="__main__":main()
