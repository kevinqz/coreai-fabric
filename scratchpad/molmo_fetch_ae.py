import urllib.request, json, os, time, numpy as np
from pathlib import Path
os.chdir("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
URL="https://huggingface.co/lerobot/MolmoAct2-LIBERO-LeRobot/resolve/main/model.safetensors"
plan=json.load(open("build/_molmoact2/ae_plan.json"))
ts=plan["tensors"]; base=plan["base"]
offs=[v["data_offsets"] for v in ts.values()]
lo=min(a for a,b in offs); hi=max(b for a,b in offs)
start=base+lo; end=base+hi; span=end-start
blob=Path("build/_molmoact2/ae_blob.bin")
def fetch(a,b,retries=6):
    for i in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(URL,headers={"Range":f"bytes={a}-{b}"}),timeout=120).read()
        except Exception as e:
            print(f"  retry {i} @{a}: {e}",flush=True); time.sleep(2*(i+1))
    raise RuntimeError(f"failed range {a}-{b}")
CHUNK=64*1024*1024
done=blob.stat().st_size if blob.exists() else 0
t=time.time()
with open(blob,"ab") as f:
    pos=start+done
    while pos<end:
        b=min(pos+CHUNK-1,end-1)
        f.write(fetch(pos,b)); f.flush()
        pos=b+1
        print(f"  {(pos-start)/1e9:.2f}/{span/1e9:.2f} GB ({(time.time()-t):.0f}s)",flush=True)
print(f"blob done {blob.stat().st_size/1e9:.2f}GB",flush=True)
from safetensors.numpy import save_file
data=blob.read_bytes()
sd={}; PRE="model.model.action_expert."
for k,v in ts.items():
    a,b=v["data_offsets"]; sh=v["shape"]
    seg=data[a-lo:b-lo]
    arr=np.frombuffer(seg,dtype="<f4")
    arr=arr.reshape(sh).copy() if sh else arr.copy()
    sd[k[len(PRE):]]=arr
save_file(sd, "build/_molmoact2/action_expert.safetensors")
print(f"reconstructed {len(sd)} tensors -> action_expert.safetensors",flush=True)
os.remove(blob); print("freed blob",flush=True)
print("FETCH+RECON DONE",flush=True)
