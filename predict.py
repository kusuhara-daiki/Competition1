import torch
import pandas as pd
import utils

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.image_process import set_img_process
from dataset.load_data import Load_Data

class Predict(Load_Data):
    """
    予測するクラス
    """
    
    def __init__(self, cfg, args):
        super.__init__(cfg, args)
        self.device = torch.device(f"cuda:{cfg.switch.device.num}" if torch.cuda.is_available() else "cpu")
        
    def make_eval_data(self): 
        eval_data = set_img_process(
            self.args, 
            self.cfg,
            self.x_test, 
        )
        
        return eval_data
        
    def dice_predict(self, eval_data):
        eval_data = torch.tensor(eval_data, dtype=torch.float32).to(self.device)
        eval_loader = DataLoader(eval_data, batch_size=32, shuffle=False)
        
        predicted_results = []
        
        with torch.no_grad():
            for images in tqdm(eval_loader, desc='predict'):
                outputs_test = self.model(images)
                predicted_test = torch.max(outputs_test, 1)[1]
                predicted_results.extend(predicted_test.cpu().numpy())
                
        result_df = pd.DataFrame({'predicted': predicted_results})
        
        return result_df
        





if __name__ == "__main__":
    
    
    args = utils.argparser()
    cfg = utils.config(args)
    
    predict = Predict(cfg, args)

    eval_data = predict.make_eval_data()
    result_df = predict.dice_predict(eval_data)
    
    result_df.to_csv(('/work/data/result/submit4.csv'), header=False)