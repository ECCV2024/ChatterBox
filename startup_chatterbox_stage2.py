import os
import cv2

PATH_TO_LLaVA = '../llava-llama-2-13b-chat-lightning-preview'

cmd = f"deepspeed --master_port 54901 train_chatterbox_stage2.py "

os.system(cmd)
