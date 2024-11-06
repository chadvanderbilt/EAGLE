import timm

def get_model():
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
