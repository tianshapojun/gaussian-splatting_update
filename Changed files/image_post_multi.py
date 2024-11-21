from PIL import Image,ImageFont,ImageDraw
import os 
from tqdm import tqdm

def concate_img(index,image_paths,width,height):
    images = [Image.open(image_path+'/%05d.png'%(index)) if idx > 0 else\
              Image.open(image_path+'/%06d.png'%(index)) for idx,image_path in enumerate(image_paths)]

    # 2. 获取图片尺寸
    image_sizes = [image.size for image in images]

    # 3. 计算长图尺寸
    long_image_width = width * max([size[0] for size in image_sizes])
    long_image_height = height * max([size[1] for size in image_sizes])

    # 4. 创建空白长图
    long_image = Image.new('RGB', (long_image_width, long_image_height))

    # 5. 拼接图片
    y_offset = 0
    font = ImageFont.truetype('/usr/share/fonts/type1/urw-base35/C059-Roman.t1',size=24)
    title_list = ['GT','Training','Customized','Customized']
    for i in range(height):
        x_offset = 0 
        for j in range(width):
            image = images[i*height+j]
            title = title_list[i*height+j]
            draw = ImageDraw.Draw(image)
            draw.text((10,10),title,font =font,fill='black')
            long_image.paste(image, (x_offset, y_offset))
            x_offset += images[i*height+j].size[0]
        y_offset += images[i*height+j].size[1]

    # 6. 保存长图
    long_image.save('/root/Codes/gaussian-splatting/output/train_06_update_0716_size_threshold2/concate_w/%05d.png'%(index))

if __name__ == "__main__":
    # 1. 加载图片
    image_paths = ['/root/Codes/gaussian-splatting/data/train_06/images',               
                #'/root/Codes/gaussian-splatting/output/train_15_update_0703/eval_stat/ours_30000',
                #'/root/Codes/gaussian-splatting/output/train_15_update_0703/eval_dyn/ours_30000',
                '/root/Codes/gaussian-splatting/output/train_06_update_0716_size_threshold2/eval/ours_50000',
                '/root/Codes/gaussian-splatting/output/train_06_update_0716_size_threshold2/eval_t/ours_50000',
                '/root/Codes/gaussian-splatting/output/train_06_update_0716_size_threshold2/eval_r/ours_50000',
                #'/root/Codes/gaussian-splatting/output/train_15_update_0703/eval_rt_10/ours_30000',
                #'/root/Codes/gaussian-splatting/output/train_15_update_0703/eval_rt20/ours_30000',
                ]
    n_files = len(os.listdir(image_paths[0]))
    l_width = 2
    l_height = 2    

    for i in tqdm(range(n_files)): 
        concate_img(i, image_paths, l_width, l_height)
    