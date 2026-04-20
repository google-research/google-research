"""This code implements a KNN service and metadata provider.

Given a query image, text, or embeddings, we find nearest neighbors and return
their indices, distances, embeddings, and other metadata.

The code below is based on https://github.com/rom1504/clip-retrieval/
"""
from typing import Callable, Dict, Any, List
from dataclasses import dataclass
import faiss
from collections import defaultdict
import json
from io import BytesIO
from PIL import Image
import numpy as np
import base64
import ssl
import os
from pathlib import Path
import pandas as pd
import urllib
import io
from shutil import copytree

import h5py
from tqdm import tqdm
import logging

LOGGER = logging.getLogger('agile_modeling')


def convert_metadata_to_base64(meta):
    """
    Converts the image at a path to the Base64 representation and sets the Base64 string to the `image`
    key in the metadata dictionary.
    If there is no `image_path` key present in the metadata dictionary, the function will have no effect.
    """
    if meta is not None and "image_path" in meta:
        path = meta["image_path"]
        if os.path.exists(path):
            img = Image.open(path)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            meta["image"] = img_str


def download_image(url):
    """Download an image from a url and return a byte stream"""
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    urllib_context = ssl.create_default_context()
    urllib_context.set_alpn_protocols(["http/1.1"])

    with urllib.request.urlopen(urllib_request, timeout=10, context=urllib_context) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)



class KnnService():
    """the knn service provides nearest neighbors given text or image"""

    def __init__(self, **kwargs):
        super().__init__()
        self.clip_resources = kwargs["clip_resources"]

    def compute_query(
        self,
        clip_resource,
        text_input,
        image_input,
        image_url_input,
        embedding_input,
        use_mclip,
    ):
        """compute the query embedding"""
        import torch  # pylint: disable=import-outside-toplevel

        if text_input is not None and text_input != "":
            if use_mclip:
                query = normalized(clip_resource.model_txt_mclip(text_input))
            else:
                text = clip_resource.tokenizer([text_input]).to(clip_resource.device)
                with torch.no_grad():
                    text_features = clip_resource.model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                query = text_features.cpu().to(torch.float32).detach().numpy()
        elif image_input is not None or image_url_input is not None:
            if image_input is not None:
                binary_data = base64.b64decode(image_input)
                img_data = BytesIO(binary_data)
            elif image_url_input is not None:
                img_data = download_image(image_url_input)
            img = Image.open(img_data)
            prepro = clip_resource.preprocess(img).unsqueeze(0).to(clip_resource.device)
            with torch.no_grad():
                image_features = clip_resource.model.encode_image(prepro)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            query = image_features.cpu().to(torch.float32).detach().numpy()
        elif embedding_input is not None:
            query = np.expand_dims(np.array(embedding_input).astype("float32"), 0)

        return query

    def hash_based_dedup(self, embeddings):
        """deduplicate embeddings based on their hash"""
        seen_hashes = set()
        to_remove = []
        for i, embedding in enumerate(embeddings):
            h = hash(np.round(embedding, 2).tobytes())
            if h in seen_hashes:
                to_remove.append(i)
                continue
            seen_hashes.add(h)

        return to_remove

    def connected_components(self, neighbors):
        """find connected components in the graph"""
        seen = set()

        def component(node):
            r = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in neighbors:
            if node not in seen:
                u.append(component(node))
        return u

    def get_non_uniques(self, embeddings, threshold=0.94):
        """find non-unique embeddings"""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)  # pylint: disable=no-value-for-parameter
        l, _, I = index.range_search(embeddings, threshold)  # pylint: disable=no-value-for-parameter,invalid-name

        same_mapping = defaultdict(list)

        # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
        for i in range(embeddings.shape[0]):
            for j in I[l[i] : l[i + 1]]:
                same_mapping[int(i)].append(int(j))

        groups = self.connected_components(same_mapping)
        non_uniques = set()
        for g in groups:
            for e in g[1:]:
                non_uniques.add(e)

        return list(non_uniques)

    def connected_components_dedup(self, embeddings):
        non_uniques = self.get_non_uniques(embeddings)
        return non_uniques

    def get_violent_items(self, safety_prompts, embeddings):
        safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
        safety_results = np.argmax(safety_predictions, axis=1)
        return np.where(safety_results == 1)[0]

    def post_filter(self, embeddings, deduplicate):
        """post filter results : dedup"""
        to_remove = set()
        if deduplicate:
#             with DEDUP_TIME.time():
            to_remove = set(self.connected_components_dedup(embeddings))

        return to_remove

    def knn_search(
        self, query, modality, num_result_ids, clip_resource, deduplicate
    ):
        """compute the knn search"""

        image_index = clip_resource.image_index
        text_index = clip_resource.text_index

        index = image_index if modality == "image" else text_index

#         with KNN_INDEX_TIME.time():
        distances, indices, embeddings = index.search_and_reconstruct(query, num_result_ids)
        results = indices[0]
                
        nb_results = np.where(results == -1)[0]

        if len(nb_results) > 0:
            nb_results = nb_results[0]
        else:
            nb_results = len(results)
        result_indices = results[:nb_results]
        result_distances = distances[0][:nb_results]
        result_embeddings = embeddings[0][:nb_results]
        result_embeddings = normalized(result_embeddings)
        local_indices_to_remove = self.post_filter(
            result_embeddings,
            deduplicate
        )
        indices_to_remove = set()
        for local_index in local_indices_to_remove:
            indices_to_remove.add(result_indices[local_index])
        indices = []
        distances = []
        embeddings = []
        for ind, distance, emb in zip(result_indices, result_distances, result_embeddings):
            if ind not in indices_to_remove:
                indices_to_remove.add(ind)
                indices.append(ind)
                distances.append(distance)
                embeddings.append(emb)

        return distances, indices, embeddings

    def map_to_metadata(self, indices, distances, embeddings, num_images, metadata_provider):
        """map the indices to the metadata"""

        results = []
        
#         with METADATA_GET_TIME.time():
        metas = metadata_provider.get(indices[:num_images])
        
        for key, (d, i, e) in enumerate(zip(distances, indices, embeddings)):
            output = {}
            meta = None if key + 1 > len(metas) else metas[key]
            convert_metadata_to_base64(meta)
            if meta is not None:
                output.update(meta_to_dict(meta))
            output["id"] = i.item()
            output["similarity"] = d.item()
            output["embedding"] = e
            results.append(output)

        return results

    def query(
        self,
        text_input=None,
        image_input=None,
        image_url_input=None,
        embedding_input=None,
        modality="image",
        num_images=100,
        num_result_ids=100,
        indice_name=None,
        use_mclip=False,
        deduplicate=True,
    ):
        """implement the querying functionality of the knn service: from text and image to nearest neighbors"""

        if text_input is None and image_input is None and image_url_input is None and embedding_input is None:
            raise ValueError("must fill one of text, image and image url input")
        if indice_name is None:
            indice_name = next(iter(self.clip_resources.keys()))

        clip_resource = self.clip_resources[indice_name]

        query = self.compute_query(
            clip_resource=clip_resource,
            text_input=text_input,
            image_input=image_input,
            image_url_input=image_url_input,
            embedding_input=embedding_input,
            use_mclip=use_mclip,
        )
        distances, indices, embeddings = self.knn_search(
            query,
            modality=modality,
            num_result_ids=num_result_ids,
            clip_resource=clip_resource,
            deduplicate=deduplicate,
        )
        if len(distances) == 0:
            return []
        results = self.map_to_metadata(
            indices, distances, embeddings, num_images, clip_resource.metadata_provider
        )

        return results

    
def meta_to_dict(meta):
    output = {}
    for k, v in meta.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif type(v).__module__ == np.__name__:
            v = v.item()
        output[k] = v
    return output

            
def load_index(path, enable_faiss_memory_mapping):
    if enable_faiss_memory_mapping:
        if os.path.isdir(path):
            return faiss.read_index(path + "/populated.index", faiss.IO_FLAG_ONDISK_SAME_DIR)
        else:
            return faiss.read_index(path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    else:
        return faiss.read_index(path)

            
class ParquetMetadataProvider:
    """The parquet metadata provider provides metadata from contiguous ids using parquet"""

    def __init__(self, parquet_folder):
        data_dir = Path(parquet_folder)
        self.metadata_df = pd.concat(
            pd.read_parquet(parquet_file) for parquet_file in sorted(data_dir.glob("*.parquet"))
        )

    def get(self, ids, cols=None):
        if cols is None:
            cols = self.metadata_df.columns.tolist()
        else:
            cols = list(set(self.metadata_df.columns.tolist()) & set(cols))

        return [self.metadata_df[i : (i + 1)][cols].to_dict(orient="records")[0] for i in ids]


def parquet_to_hdf5(parquet_folder, output_hdf5_file, columns_to_return=["url", "image_path", "caption", "NSFW"]):
    """this convert a collection of parquet file to an hdf5 file"""
    f = h5py.File(output_hdf5_file, "w")
    data_dir = Path(parquet_folder)
    ds = f.create_group("dataset")
    for parquet_files in tqdm(sorted(data_dir.glob("*.parquet"))):
        df = pd.read_parquet(parquet_files)
        for k in df.keys():
            if k not in columns_to_return:
                continue
            col = df[k]
            if col.dtype in ("float64", "float32"):
                col = col.fillna(0.0)
            if col.dtype in ("int64", "int32"):
                col = col.fillna(0)
            if col.dtype == "object":
                col = col.fillna("")
                col = col.str.replace("\x00", "", regex=False)
            z = col.to_numpy()
            if k not in ds:
                ds.create_dataset(k, data=z, maxshape=(None,), compression="gzip")
            else:
                prevlen = len(ds[k])
                ds[k].resize((prevlen + len(z),))
                ds[k][prevlen:] = z

    del ds
    f.close()


class Hdf5MetadataProvider:
    """The hdf5 metadata provider provides metadata from contiguous ids using hdf5"""

    def __init__(self, hdf5_file):
        f = h5py.File(hdf5_file, "r")
        self.ds = f["dataset"]

    def get(self, ids, cols=None):
        """implement the get method from the hdf5 metadata provide, get metadata from ids"""
        items = [{} for _ in range(len(ids))]
        if cols is None:
            cols = self.ds.keys()
        else:
            cols = list(self.ds.keys() & set(cols))
        for k in cols:
            for i, e in enumerate(ids):
                items[i][k] = self.ds[k][e]
        return items


def load_metadata_provider(
    indice_folder, enable_hdf5, image_index
):
    """load the metadata provider"""
    parquet_folder = indice_folder + "/metadata"
    if enable_hdf5:
        hdf5_path = indice_folder + "/metadata.hdf5"
        if not os.path.exists(hdf5_path):
            parquet_to_hdf5(parquet_folder, hdf5_path)
        metadata_provider = Hdf5MetadataProvider(hdf5_path)
    else:
        metadata_provider = ParquetMetadataProvider(parquet_folder)

    return metadata_provider




@dataclass
class ClipResource:
    """the resource for clip : model, index, options"""

    device: str
    model: Any
    preprocess: Callable
    tokenizer: Callable
    model_txt_mclip: Any
    metadata_provider: Any
    image_index: Any
    text_index: Any


@dataclass
class ClipOptions:
    """the options for clip"""

    indice_folder: str
    clip_model: str
    enable_hdf5: bool
    enable_faiss_memory_mapping: bool
    enable_mclip_option: bool
    use_jit: bool


def dict_to_clip_options(d, clip_options):
    return ClipOptions(
        indice_folder=d["indice_folder"] if "indice_folder" in d else clip_options.indice_folder,
        clip_model=d["clip_model"] if "clip_model" in d else clip_options.clip_model,
        enable_hdf5=d["enable_hdf5"] if "enable_hdf5" in d else clip_options.enable_hdf5,
        enable_faiss_memory_mapping=d["enable_faiss_memory_mapping"]
        if "enable_faiss_memory_mapping" in d
        else clip_options.enable_faiss_memory_mapping,
        enable_mclip_option=d["enable_mclip_option"]
        if "enable_mclip_option" in d
        else clip_options.enable_mclip_option,
        use_jit=d["use_jit"] if "use_jit" in d else clip_options.use_jit,
    )



def load_clip_index(clip_options):
    """load the clip index"""
    import torch  # pylint: disable=import-outside-toplevel
    from load_clip import load_clip, get_tokenizer  # pylint: disable=import-outside-toplevel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip(clip_options.clip_model, use_jit=clip_options.use_jit, device=device)

    tokenizer = get_tokenizer(clip_options.clip_model)

    if clip_options.enable_mclip_option:
        model_txt_mclip = load_mclip(clip_options.clip_model)
    else:
        model_txt_mclip = None

    image_present = os.path.exists(clip_options.indice_folder + "/image.index")
    text_present = os.path.exists(clip_options.indice_folder + "/text.index")

    LOGGER.info("loading indices...")
    image_index = (
        load_index(clip_options.indice_folder + "/image.index", clip_options.enable_faiss_memory_mapping)
        if image_present
        else None
    )
    text_index = (
        load_index(clip_options.indice_folder + "/text.index", clip_options.enable_faiss_memory_mapping)
        if text_present
        else None
    )

    LOGGER.info("loading metadata...")

    metadata_provider = load_metadata_provider(
        clip_options.indice_folder,
        clip_options.enable_hdf5,
        image_index,
    )

    return ClipResource(
        device=device,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        model_txt_mclip=model_txt_mclip,
        metadata_provider=metadata_provider,
        image_index=image_index,
        text_index=text_index,
    )



def load_clip_indices(
    indices_paths,
    clip_options,
) -> Dict[str, ClipResource]:
    """This load clips indices from disk"""
    LOGGER.info("loading clip...")

    with open(indices_paths, "r", encoding="utf-8") as f:
        indices = json.load(f)

    clip_resources = {}

    for name, indice_value in indices.items():
        # if indice_folder is a string
        if isinstance(indice_value, str):
            clip_options = dict_to_clip_options({"indice_folder": indice_value}, clip_options)
        elif isinstance(indice_value, dict):
            clip_options = dict_to_clip_options(indice_value, clip_options)
        else:
            raise ValueError("Unknown type for indice_folder")
        clip_resources[name] = load_clip_index(clip_options)

    return clip_resources

def create(
    indices_paths="indices.json",
    enable_hdf5=False,
    enable_faiss_memory_mapping=True,
    enable_mclip_option=False, # TODO (@evendrow): set this back to true
    clip_model="ViT-B/32",
    use_jit=True,
):
    
    clip_resources = load_clip_indices(
            indices_paths=indices_paths,
            clip_options=ClipOptions(
                indice_folder="./index",
                clip_model=clip_model,
                enable_hdf5=enable_hdf5,
                enable_faiss_memory_mapping=enable_faiss_memory_mapping,
                enable_mclip_option=enable_mclip_option,
                use_jit=use_jit,
            ),
        )
    
    knn_service = KnnService(clip_resources=clip_resources)
    return knn_service
