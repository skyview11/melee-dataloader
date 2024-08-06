import os
from slp2memmap import Slp2MemmapConverter
from tqdm import tqdm

class SLPPreprocessor:
    def __init__(self, slp_base_path, memmap_base_path):
        """convert slp file to numpy memmap format

        Args:
            slp_base_path (str): base path of flattened slp files
            memmap_base_path (str): base path of converted memmap file
        """
        self.slp_base_path = slp_base_path
        self.mm_base_path = memmap_base_path
        self.converter = Slp2MemmapConverter()
    def process(self):
        """convert 
        """
        file_names = [name.split(".")[0] 
                      for name in os.listdir(self.slp_base_path)]
        
        slp_abs_paths = [os.path.join(self.slp_base_path, f"{file_name}.slp") 
                         for file_name in file_names]
        
        mm_abs_paths = [os.path.join(self.mm_base_path, f"{file_name}.dat") 
                         for file_name in file_names]
        
        config_base_path = os.path.join(self.mm_base_path, 'config')
        os.makedirs(config_base_path, exist_ok=True)
        config_abs_paths = [os.path.join(config_base_path, f"{file_name}_conf.json")
                            for file_name in file_names]
        pbar = tqdm(total=len(file_names))
        for slp_file, mm_file, conf_file in zip(slp_abs_paths, mm_abs_paths, config_abs_paths):
            self.converter.convert(slp_file, mm_file, conf_file)
            pbar.update(1)
        pbar.close()
def main(args):
    """main function to convert slp to memmap, only used when run this file
    """
    
    preprocessor = SLPPreprocessor(args.source_path, args.target_path)
    preprocessor.process()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, default="/root/data/jay/training_data", help="source path of flattened slp dataset")
    parser.add_argument("--target_path", type=str, default="/root/data/skyview/mm", help="target path of flattened memmap dataset")
    args = parser.parse_args()
    main(args)
    
    