from PIL import Image, ImageDraw, ImageFont
import os

def combine_resized_images_with_labels(image_path1, image_path2, output_path):
    # 打开两张图片
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # 调整图片尺寸
    img1 = img1.resize((224, 224))
    img2 = img2.resize((224, 224))

    # 创建新图片，宽度为两张图片之和，高度稍高一些以容纳文字
    new_width = 224 * 2
    new_height = 224 + 50  # 假设白边高度为50像素
    new_img = Image.new('RGB', (new_width, new_height), 'white')

    # 将调整大小后的图片粘贴到新图片上
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (224, 0))

    # 在图片下方添加文字
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.truetype("/data/zhangyichi/Trustworthy-MLLM/data/fairness/Arial.ttf", 36)  # 选择字体和大小
    draw.text((224 // 2 - 20, 224 + 10), "A", fill="black", font=font)
    draw.text((224 + 224 // 2 - 20, 224 + 10), "B", fill="black", font=font)

    # 保存新图片
    new_img.save(output_path)

img_a_path = "/data/zhangyichi/Trustworthy-MLLM/data/fairness/subjective_choice_images/choice_a_images"
img_b_path = "/data/zhangyichi/Trustworthy-MLLM/data/fairness/subjective_choice_images/choice_b_images"
img_mixed_path = "/data/zhangyichi/Trustworthy-MLLM/data/fairness/vision_subjective_choice_images"

# 使用示例
for img in os.listdir("/data/zhangyichi/Trustworthy-MLLM/data/fairness/subjective_choice_images/choice_a_images"):
    combine_resized_images_with_labels(os.path.join(img_a_path, img), os.path.join(img_b_path, img), os.path.join(img_mixed_path, img))
