from PIL import Image,ImageFont,ImageDraw
import os 
from tqdm import tqdm

def concate_img(index,image_paths):
    images = [Image.open(image_path+'/%05d.png'%(index)) if idx > 0 else\
              Image.open(image_path+'/%05d.png'%(index)) for idx,image_path in enumerate(image_paths)]
    #images = [Image.open(image_path+'/%05d.png'%(index)) for image_path in image_paths]

    # 2. 获取图片尺寸
    image_sizes = [image.size for image in images]

    # 3. 计算长图尺寸
    long_image_width = max([size[0] for size in image_sizes])
    long_image_height = sum([size[1] for size in image_sizes])

    # 4. 创建空白长图
    long_image = Image.new('RGB', (long_image_width, long_image_height))

    # 5. 拼接图片
    y_offset = 0
    font = ImageFont.truetype('/usr/share/fonts/type1/urw-base35/C059-Roman.t1',size=24)
    #title_list = ['GT','Training','Customized','Customized']
    title_list = ['Original','Optimized']
    for idx,image in enumerate(images):
        title = title_list[idx]
        draw = ImageDraw.Draw(image)
        draw.text((10,10),title,font =font,fill='black')
        long_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    # 6. 保存长图
    long_image.save('/root/Codes/gaussian-splatting/output/train_06_update_0801_rot/compare/%05d.png'%(index))

if __name__ == "__main__":
    # 1. 加载图片
    image_paths = [#'/root/Codes/gaussian-splatting/data/train_06/images',               
                #'/root/Codes/gaussian-splatting/output/train_06_update_0716_size_threshold2/eval/ours_50000',
                #'/root/Codes/gaussian-splatting/output/train_06_update_0716_size_threshold2/eval_t/ours_50000',
                #'/root/Codes/gaussian-splatting/output/train_06_update_0716_size_threshold2/eval_r/ours_50000',
                '/root/Codes/gaussian-splatting/output/train_06_update_0716_size_threshold2/eval_r/ours_50000',
                '/root/Codes/gaussian-splatting/output/train_06_update_0801_rot/eval_r/ours_50000'
                ]
    n_files = len(os.listdir(image_paths[0]))
    
    for i in tqdm(range(n_files)): 
        concate_img(i, image_paths)
    