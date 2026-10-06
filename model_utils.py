
import os, re, json, zipfile, urllib.request, socket
from urllib.parse import urlparse, urljoin
import numpy as np
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup

try:
    import whois
except Exception:
    whois = None
try:
    import dns.resolver
except Exception:
    dns = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

BASE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(BASE,"data"); MODEL_DIR=os.path.join(BASE,"models"); RESULTS_DIR=os.path.join(BASE,"results")
for d in (DATA_DIR,MODEL_DIR,RESULTS_DIR): os.makedirs(d,exist_ok=True)
DATA_URL="https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
ZIP_PATH=os.path.join(DATA_DIR,"phishing_websites.zip")
CSV_PATH=os.path.join(DATA_DIR,"phishing.csv")
BUNDLE_PATH=os.path.join(MODEL_DIR,"aipdnet_bundle.pkl")

FEATURES=[
"having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol",
"double_slash_redirecting","Prefix_Suffix","having_Sub_Domain",
"SSLfinal_State","Domain_registeration_length","Favicon","port",
"HTTPS_token","Request_URL","URL_of_Anchor","Links_in_tags","SFH",
"Submitting_to_email","Abnormal_URL","Redirect","on_mouseover",
"RightClick","popUpWindow","Iframe","age_of_domain","DNSRecord",
"web_traffic","Page_Rank","Google_Index","Links_pointing_to_page",
"Statistical_report"
]

def _parse_arff(path):
    rows=[]; in_data=False
    with open(path,"r",errors="ignore") as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith("%"): continue
            if s.lower().startswith("@data"): in_data=True; continue
            if in_data:
                parts=[p.strip() for p in s.split(",")]
                if len(parts)==len(FEATURES)+1:
                    try: rows.append([int(float(p)) for p in parts])
                    except ValueError: pass
    if not rows: raise ValueError("No data rows parsed from "+path)
    return pd.DataFrame(rows,columns=FEATURES+["Result"])

def _from_files(folder):
    arffs=[]; csvs=[]
    for base,_,fs in os.walk(folder):
        for f in fs:
            p=os.path.join(base,f)
            if p==CSV_PATH: continue
            if f.lower().endswith(".arff"): arffs.append(p)
            elif f.lower().endswith(".csv"): csvs.append(p)
    arffs.sort(key=lambda p:("old" in p.lower(),len(p)))
    for p in arffs:
        try: return _parse_arff(p)
        except Exception: continue
    for p in csvs:
        try:
            d=pd.read_csv(p)
            if d.shape[1]==len(FEATURES)+2 and str(d.columns[0]).lower() in ("index","id"): d=d.iloc[:,1:]
            if d.shape[1]==len(FEATURES)+1:
                d.columns=FEATURES+["Result"]; return d
        except Exception: continue
    return None

def _download_uci():
    urllib.request.urlretrieve(DATA_URL,ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH,"r") as z: z.extractall(DATA_DIR)
    return _from_files(DATA_DIR)

def _download_ucimlrepo():
    from ucimlrepo import fetch_ucirepo
    d=fetch_ucirepo(id=327)
    df=pd.concat([d.data.features.reset_index(drop=True),d.data.targets.reset_index(drop=True)],axis=1)
    df=df.iloc[:,-(len(FEATURES)+1):].copy(); df.columns=FEATURES+["Result"]
    return df.astype(str).apply(pd.to_numeric)

def _download_openml():
    from sklearn.datasets import fetch_openml
    d=fetch_openml(data_id=4534,as_frame=True).frame
    d=d.iloc[:,-(len(FEATURES)+1):].copy(); d.columns=FEATURES+["Result"]
    return d.astype(str).apply(pd.to_numeric)

def download_dataset():
    if os.path.exists(CSV_PATH): return
    df=_from_files(DATA_DIR); errs=[]
    for fn in (_download_uci,_download_ucimlrepo,_download_openml):
        if df is not None: break
        try: df=fn()
        except Exception as e: errs.append(fn.__name__+": "+str(e))
    if df is None or df.empty:
        raise RuntimeError("Could not obtain the UCI Phishing Websites dataset ("+"; ".join(errs)+"). Download 'Training Dataset.arff' from https://archive.ics.uci.edu/dataset/327/phishing+websites and put it in the 'data' folder, then retry.")
    df.to_csv(CSV_PATH,index=False)

def load_dataset():
    download_dataset()
    df=pd.read_csv(CSV_PATH)
    df.columns=[str(c).strip() for c in df.columns]
    if df.shape[1]==len(FEATURES)+1: df.columns=FEATURES+["Result"]
    if "Result" not in df.columns: raise ValueError("Result target column was not found.")
    return df

def train_models():
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    raw=load_dataset(); df=raw.drop_duplicates().dropna()
    X=df[FEATURES].apply(pd.to_numeric,errors="coerce").fillna(0)
    y=pd.to_numeric(df["Result"],errors="coerce").map(lambda v:1 if v==-1 else 0)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    stats={"raw_instances":int(len(raw)),"unique_instances":int(len(df)),"phishing":int(y.sum()),"legitimate":int((y==0).sum()),"train":int(len(X_train)),"test":int(len(X_test)),"features":len(FEATURES)}
    json.dump(stats,open(os.path.join(RESULTS_DIR,"dataset_stats.json"),"w"),indent=2); print("Dataset:",stats)
    scaler=StandardScaler()
    Xtr=scaler.fit_transform(X_train); Xte=scaler.transform(X_test)
    rf=RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced")
    xgb=XGBClassifier(n_estimators=250,max_depth=6,learning_rate=.08,
                      subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=42)
    lgbm=LGBMClassifier(n_estimators=250,learning_rate=.08,num_leaves=31,verbosity=-1,random_state=42)
    models={"Random Forest":rf,"XGBoost":xgb,"LightGBM":lgbm}; rows=[]
    for name,m in models.items():
        m.fit(Xtr,y_train); pred=m.predict(Xte); prob=m.predict_proba(Xte)[:,1]
        rows.append({"Model":name,"Accuracy":accuracy_score(y_test,pred),
                     "Precision":precision_score(y_test,pred,zero_division=0),
                     "Recall":recall_score(y_test,pred,zero_division=0),
                     "F1":f1_score(y_test,pred,zero_division=0),
                     "ROC-AUC":roc_auc_score(y_test,prob)})
    ensemble=VotingClassifier(estimators=[("rf",rf),("xgb",xgb),("lgbm",lgbm)],voting="soft")
    ensemble.fit(Xtr,y_train); pred=ensemble.predict(Xte); prob=ensemble.predict_proba(Xte)[:,1]
    rows.append({"Model":"AIPDNet Ensemble","Accuracy":accuracy_score(y_test,pred),
                 "Precision":precision_score(y_test,pred,zero_division=0),
                 "Recall":recall_score(y_test,pred,zero_division=0),
                 "F1":f1_score(y_test,pred,zero_division=0),
                 "ROC-AUC":roc_auc_score(y_test,prob)})
    metrics=pd.DataFrame(rows); metrics.to_csv(os.path.join(RESULTS_DIR,"model_comparison.csv"),index=False)
    try:
        import shap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        explainer=shap.TreeExplainer(rf)
        sample=X_test.sample(min(500,len(X_test)),random_state=42)
        vals=explainer.shap_values(scaler.transform(sample))
        vals=vals[1] if isinstance(vals,list) else (vals[:,:,1] if np.ndim(vals)==3 else vals)
        plt.figure(figsize=(9,6))
        shap.summary_plot(vals,sample,feature_names=FEATURES,show=False,max_display=15)
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR,"shap_summary.png"),dpi=180,bbox_inches="tight"); plt.close()
    except Exception as e: print("SHAP plot skipped:",e)
    bundle={"scaler":scaler,"ensemble":ensemble,"rf":rf,"xgb":xgb,"lgbm":lgbm,
            "feature_names":FEATURES,"metrics":metrics,"stats":stats}
    joblib.dump(bundle,BUNDLE_PATH)
    return bundle

def load_or_train(force=False):
    return train_models() if force or not os.path.exists(BUNDLE_PATH) else joblib.load(BUNDLE_PATH)

def get_feature_names(bundle=None): return FEATURES

SHORTENERS={"bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly","cutt.ly","rebrand.ly"}

def _fetch_page(url):
    headers={"User-Agent":"Mozilla/5.0 AIPDNet Academic Security Scanner"}
    try:
        r=requests.get(url,headers=headers,timeout=8,allow_redirects=True,verify=True)
        return r, BeautifulSoup(r.text,"html.parser")
    except Exception:
        return None,None

def _domain_age_days(host):
    if not whois: return 0
    try:
        socket.setdefaulttimeout(8)
        w=whois.whois(host)
        created=w.creation_date
        if isinstance(created,list): created=created[0]
        if created:
            from datetime import datetime, timezone
            if created.tzinfo is None: created=created.replace(tzinfo=timezone.utc)
            return max(0,(datetime.now(timezone.utc)-created).days)
    except Exception: pass
    return 0

def extract_url_features(url,feature_names=FEATURES):
    original=url.strip()
    if not re.match(r"^https?://",original,re.I): original="http://"+original
    p=urlparse(original); host=p.hostname or ""; host=host.lower()
    full=original.lower(); path=p.path or ""
    f={}; info={}
    f["having_IP_Address"]=-1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$",host) else 1
    f["URL_Length"]=-1 if len(full)>=75 else (0 if len(full)>=54 else 1)
    f["Shortining_Service"]=-1 if host in SHORTENERS or any(host.endswith("."+s) for s in SHORTENERS) else 1
    f["having_At_Symbol"]=-1 if "@" in full else 1
    f["double_slash_redirecting"]=-1 if "//" in full[8:] else 1
    f["Prefix_Suffix"]=-1 if "-" in host else 1
    labels=host.split(".") if host else []
    f["having_Sub_Domain"]=-1 if len(labels)>=4 else (0 if len(labels)==3 else 1)
    f["SSLfinal_State"]=1 if p.scheme=="https" else -1
    f["Domain_registeration_length"]=0
    f["Favicon"]=0; f["port"]=-1 if p.port not in (None,80,443) else 1
    f["HTTPS_token"]=-1 if "https" in host else 1

    r,soup=_fetch_page(original)
    if r is not None:
        final=urlparse(r.url)
        final_host=(final.hostname or "").lower()
        f["Redirect"]=-1 if final_host!=host else 1
        f["Abnormal_URL"]=1 if final_host else -1
        links=soup.find_all("a",href=True) if soup else []
        imgs=soup.find_all(["img","script"],src=True) if soup else []
        anchors=0
        external_anchors=0
        for a in links:
            href=a.get("href","")
            if href.startswith(("http://","https://")):
                anchors+=1
                try:
                    if (urlparse(href).hostname or "").lower()!=host: external_anchors+=1
                except Exception: pass
        f["URL_of_Anchor"]=-1 if anchors and external_anchors/anchors>.67 else (1 if anchors else 0)
        f["Request_URL"]=-1 if imgs and sum(1 for x in imgs if urlparse(urljoin(original,x.get("src",""))).hostname not in (None,host))/len(imgs)>.5 else (1 if imgs else 0)
        f["Links_in_tags"]=1 if soup and (soup.find_all("link") or soup.find_all("meta")) else 0
        text=soup.get_text(" ",strip=True).lower() if soup else ""
        f["Submitting_to_email"]=-1 if soup and soup.find("form") and ("mailto:" in str(soup).lower() or "email" in str(soup).lower()) else 1
        f["SFH"]=-1 if soup and any(form.get("action","").startswith(("http://","https://")) for form in soup.find_all("form")) else 1
        f["Favicon"]=1 if soup and soup.find("link",rel=lambda x:x and "icon" in str(x).lower()) else 0
        f["on_mouseover"]=-1 if soup and "onmouseover" in str(soup).lower() else 1
        f["RightClick"]=-1 if soup and ("contextmenu" in str(soup).lower() or "event.button" in str(soup).lower()) else 1
        f["popUpWindow"]=-1 if soup and any(x in str(soup).lower() for x in ["window.open(","popup"]) else 1
        f["Iframe"]=-1 if soup and soup.find("iframe") else 1
        f["Google_Index"]=0
        info["HTTP status"]=r.status_code
        info["Final URL"]=r.url
        info["Redirected"]=final_host!=host
        info["External anchors"]=f["URL_of_Anchor"]
    else:
        for k in ["Request_URL","URL_of_Anchor","Links_in_tags","SFH","Submitting_to_email",
                  "Redirect","on_mouseover","RightClick","popUpWindow","Iframe"]:
            f[k]=0
        f["Abnormal_URL"]=1

    age=_domain_age_days(host)
    f["age_of_domain"]=1 if age>=180 else (-1 if age and age<180 else 0)
    try:
        socket.gethostbyname(host); f["DNSRecord"]=1
    except Exception: f["DNSRecord"]=-1
    f["web_traffic"]=0; f["Page_Rank"]=0; f["Google_Index"]=0
    f["Links_pointing_to_page"]=0; f["Statistical_report"]=0

    info.update({
        "Hostname":host,"URL length":len(full),"HTTPS":p.scheme=="https",
        "Domain age (days)":age,"DNS resolves":f["DNSRecord"]==1,
        "Subdomains":max(0,len(labels)-2),"Has @":"@" in full,
        "Has '-' in domain":"-" in host
    })
    vector=np.array([[f.get(k,0) for k in feature_names]],dtype=float)
    return vector,info

def predict_url(bundle,features):
    scaled=bundle["scaler"].transform(pd.DataFrame(features,columns=bundle["feature_names"]))
    pred=int(bundle["ensemble"].predict(scaled)[0])
    prob=float(bundle["ensemble"].predict_proba(scaled)[0,1])
    return pred,prob

def calculate_risk(phishing_probability):
    score=round(float(phishing_probability)*100,2)
    if score<=30: level="SAFE"
    elif score<=60: level="SUSPICIOUS"
    else: level="HIGH RISK"
    return score,level
