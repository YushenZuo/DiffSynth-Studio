from .operators import *
import torch, json, pandas
from pathlib import Path
from collections.abc import Mapping


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat

    def get_stable_id(self, data_id):
        """Stable id for the sample at index data_id (no shuffle / order change). Used for validation alignment."""
        if self.load_from_cache:
            path = self.cached_data[data_id % len(self.cached_data)]
            return (Path(path).stem, Path(path).parent.name)
        row = self.data[data_id % len(self.data)]
        return row.get("clip_id", row.get("id", data_id))

    def get_indices_for_ids(self, ids):
        """Return list of indices whose get_stable_id(i) is in ids (first occurrence per id)."""
        id2idx = {}
        n = len(self.data) if not self.load_from_cache else len(self.cached_data)
        for i in range(n):
            sid = self.get_stable_id(i)
            if sid not in id2idx:
                id2idx[sid] = i
        return [id2idx[sid] for sid in ids if sid in id2idx]

    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True



class UnifiedDataset2(UnifiedDataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        load_metadata_and_cache=False,          # load base cached data, process additional new metadata
        load_base_and_added_cache=False,        # load base cahced data, additional cached data
        added_cache_path=None,
    ):
        self.load_metadata_and_cache = load_metadata_and_cache
        self.load_base_and_added_cache = load_base_and_added_cache
        self.added_cache_path = added_cache_path
        self.clip_id_map = {}

        if self.load_base_and_added_cache and not self.added_cache_path:
            raise ValueError("added_cache_path must be provided if load_base_and_added_cache is True")

        super().__init__(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=repeat,
            data_file_keys=data_file_keys,
            main_data_operator=main_data_operator,
            special_operator_map=special_operator_map,
        )

    def _merge_samples(self, base_sample, added_sample):
        if not isinstance(base_sample, (list, tuple)):
            if isinstance(base_sample, dict) and isinstance(added_sample, dict):
                return {**base_sample, **added_sample}
            return base_sample
        
        base_list = list(base_sample)
        for i, added_item in enumerate(added_sample):
            if isinstance(added_item, Mapping) and i < len(base_list):
                if isinstance(base_list[i], dict):
                    base_list[i] = {**base_list[i], **added_item}
        return tuple(base_list)

    def _get_common_metadata(self, path):
        path_obj = Path(path)
        clip_id = path_obj.stem
        process_index = path_obj.parent.name
        return clip_id, process_index

    def load_metadata(self, metadata_path):
        if self.load_metadata_and_cache or self.load_base_and_added_cache:
            self.search_for_cached_data_files(self.base_path)

            if metadata_path:
                super().load_metadata(metadata_path)
                self.clip_id_map = {str(item['clip_id']): item for item in self.data if 'clip_id' in item}
                print(f"[INFO] Built clip_id index map with {len(self.clip_id_map)} entries.")
            elif self.load_metadata_and_cache:
                raise RuntimeError("metadata_path is required for load_metadata_and_cache=True.")
            return

        super().load_metadata(metadata_path)

    def __getitem__(self, data_id, __depth=0):
        if __depth > 10:
            raise RuntimeError(f"Maximum retry depth reached at index {data_id}")

        try:
            # TODO: detect shape
            if self.load_metadata_and_cache or self.load_base_and_added_cache:
                if not self.cached_data:
                    raise RuntimeError("No cached data files available.")
                
                path = self.cached_data[data_id % len(self.cached_data)]
                cached_sample = self.cached_data_operator(path)

                clip_id, proc_idx = self._get_common_metadata(path)

                if self.load_base_and_added_cache:
                    added_path = os.path.join(self.added_cache_path, proc_idx, f"{clip_id}.pth")
                    if os.path.exists(added_path):
                        added_sample = self.cached_data_operator(added_path)
                        cached_sample = self._merge_samples(cached_sample, added_sample)
                    if not self.load_metadata_and_cache:
                        return cached_sample
                
                if self.load_metadata_and_cache:
                    metadata = self.clip_id_map.get(clip_id)
                    if metadata is None:
                        raise RuntimeError(f"Metadata missing for clip_id: {clip_id}")
                    
                    metadata_sample = metadata.copy()
                    for key in self.data_file_keys:
                        if key in metadata_sample:
                            op = self.special_operator_map.get(key, self.main_data_operator)
                            metadata_sample[key] = op(metadata_sample[key])
                    metadata_sample.update({'clip_id': clip_id, 'process_index': proc_idx})
                    return metadata_sample, cached_sample

            return super().__getitem__(data_id)

        except Exception as e:
            print(f"[WARN] Sample {data_id} failed: {e}. Retrying next...")
            return self.__getitem__((data_id + 1) % len(self), __depth + 1)