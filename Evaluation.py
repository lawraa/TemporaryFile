import os
import sys
sys.path.append('Dice_Coefficient')
sys.path.append('Average_Precision')

from Dice_Coefficient import dice_test
from Average_Precision import MAP
import IoU

def eval(path="Output_Data"):
    
    Dice_Coefficient = dice_test.dice()
    os.chdir('../')
    
    Average_Precision = MAP.map_cal(path)
    print(Average_Precision)
    
    avg_IoU = IoU.get_IoU_for_dir(path)
    
    
    return [Dice_Coefficient, Average_Precision, avg_IoU]




if __name__ == "__main__":
    [Dice_Coefficient, Average_Precision, avg_IoU]=eval()
    print('\n\n\n\n----------------------------------------------result----------------------------------------------')
    print("Dice_Coefficient = ",Dice_Coefficient)
    print("Average_Precision = ",Average_Precision)
