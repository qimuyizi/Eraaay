import os
import shutil
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
import logging

# 初始化设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device.upper()}")

# 创建输出目录结构
def create_folders(root_folder):
    folders = ['人物', '证件', '风景']
    for folder in folders:
        path = os.path.join(root_folder, folder)
        os.makedirs(path, exist_ok=True)
    return root_folder

# 场景分类模型（使用Places365）
def load_scene_model():
    model = models.resnet18(num_classes=365)
    model_path = 'resnet18_places365.pth.tar'
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model.eval().to(device)

# 加载场景分类标签
def load_scene_classes():
    classes = []
    with open('categories_places365.txt', 'r') as f:
        for line in f:
            classes.append(line.strip().split(' ')[0][3:])
    return classes

# 人脸处理模型
mtcnn = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 图像预处理
scene_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 证件照判断标准
def is_id_photo(img, boxes):
    if len(boxes) != 1:
        return False
    
    # 人脸区域占比
    img_w, img_h = img.size
    x1, y1, x2, y2 = boxes[0]
    face_area = (x2 - x1) * (y2 - y1)
    if face_area / (img_w * img_h) < 0.3:
        return False
    
    # 背景检测
    corners = [
        img.crop((0, 0, img_w//4, img_h//4)),
        img.crop((3*img_w//4, 0, img_w, img_h//4)),
        img.crop((0, 3*img_h//4, img_w//4, img_h)),
        img.crop((3*img_w//4, 3*img_h//4, img_w, img_h))
    ]
    variances = [np.array(c).var() for c in corners]
    if max(variances) > 500:
        return False
    
    return True

def move_file(src, dst_folder, filename):
    """移动文件到目标文件夹，处理重名文件"""
    dst = os.path.join(dst_folder, filename)
    base, ext = os.path.splitext(filename)
    counter = 1
    
    while os.path.exists(dst):
        new_name = f"{base}_{counter}{ext}"
        dst = os.path.join(dst_folder, new_name)
        counter += 1
        
    shutil.move(src, dst)
    logging.info(f"已移动文件: {src} -> {dst}")

def create_character_folder(root_folder, character_name):
    """创建角色文件夹"""
    path = os.path.join(root_folder, '人物', character_name)
    os.makedirs(path, exist_ok=True)
    return path

# 方案1：通过文件名识别角色
def get_characters_from_filename(filename):
    """从文件名获取角色名称
    文件名格式示例：'宫水三叶_立花泷.jpg' 或 '[宫水三叶,立花泷]照片.jpg'"""
    name = os.path.splitext(filename)[0]  # 去除扩展名
    
    # 支持多种分隔符
    for separator in ['_', ',', '，', '&', '+', ' ']:
        if separator in name:
            return [char.strip('[]【】()（）') for char in name.split(separator)]
    
    return [name.strip('[]【】()（）')]  # 如果没有分隔符，将整个文件名作为角色名

# 方案2：通过用户交互输入角色名
def get_characters_from_user(filepath):
    """显示图片并让用户输入角色名称"""
    try:
        # 显示图片（如果在终端环境下可以使用其他查看器）
        img = Image.open(filepath)
        img.show()
        
        # 获取用户输入
        print(f"\n请输入图片中的角色名称（多个角色用逗号分隔）：")
        characters = input().strip()
        if characters:
            return [char.strip() for char in characters.split(',')]
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
    return []

def classify_photos(input_folder, output_folder, use_filename=True):
    """对照片进行分类
    use_filename: True表示使用文件名识别角色，False表示使用用户输入"""
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            continue
            
        filepath = os.path.join(input_folder, filename)
        
        try:
            # 获取角色列表
            if use_filename:
                characters = get_characters_from_filename(filename)
            else:
                characters = get_characters_from_user(filepath)
            
            if characters:  # 如果有角色信息
                for character in characters:
                    if character:  # 确保角色名不为空
                        # 为每个角色创建文件夹并复制图片
                        char_folder = create_character_folder(output_folder, character)
                        shutil.copy2(filepath, os.path.join(char_folder, filename))
                continue
            
            # 如果没有识别到角色，按原有逻辑处理
            # ... existing code ...
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue

# ...（保持原有的get_folder_path和get_output_folder函数不变）
# ... existing code ...

def get_folder_path():
    """获取并验证文件夹路径"""
    while True:
        folder_path = input("请输入需要整理的照片文件夹路径（例如：C:\\Photos）：").strip()
        
        # 处理用户输入的引号
        folder_path = folder_path.strip('"\'')
        
        # 检查路径是否存在
        if not os.path.exists(folder_path):
            print(f"错误：文件夹 '{folder_path}' 不存在，请重新输入")
            continue
            
        # 检查是否为文件夹
        if not os.path.isdir(folder_path):
            print(f"错误：'{folder_path}' 不是一个文件夹，请重新输入")
            continue
            
        # 检查是否有权限访问
        try:
            test_file = os.path.join(folder_path, 'test_permission')
            with open(test_file, 'w') as f:
                pass
            os.remove(test_file)
        except Exception as e:
            print(f"错误：没有文件夹的访问权限，请检查权限或选择其他文件夹")
            continue
            
        return folder_path

def get_output_folder(input_folder):
    """获取输出文件夹路径"""
    while True:
        choice = input("请选择分类后的照片存放位置：\n1. 在原文件夹中创建分类文件夹\n2. 选择其他位置\n请输入选项（1/2）：").strip()
        
        if choice == '1':
            return input_folder
        elif choice == '2':
            output_path = input("请输入存放位置的完整路径：").strip().strip('"\'')
            
            # 如果路径不存在，询问是否创建
            if not os.path.exists(output_path):
                create = input(f"文件夹 '{output_path}' 不存在，是否创建？(y/n)：").lower()
                if create == 'y':
                    try:
                        os.makedirs(output_path)
                        return output_path
                    except Exception as e:
                        print(f"创建文件夹失败：{str(e)}")
                        continue
                else:
                    continue
            
            if os.path.isdir(output_path):
                return output_path
            else:
                print("错误：指定的路径不是一个文件夹，请重新输入")
        else:
            print("无效的选项，请重新输入")

if __name__ == '__main__':
    print("欢迎使用照片智能分类工具！")
    
    # 获取输入文件夹路径
    input_folder = get_folder_path()
    
    # 获取输出位置
    output_folder = get_output_folder(input_folder)
    
    print(f"\n开始处理照片...")
    print(f"输入文件夹：{input_folder}")
    print(f"输出位置：{output_folder}")
    
    try:
        # 选择使用文件名方式还是用户输入方式
        use_filename = True  # 改为False则使用用户输入方式
        classify_photos(input_folder, output_folder, use_filename)
        print("\n✨ 分类完成！")
        print(f"请在以下位置查看分类结果：{output_folder}")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误：{str(e)}")
    
    input("\n按回车键退出程序...")

# 需要修改classify_photos函数接收output_folder参数
def classify_photos(input_folder, output_folder):
    scene_model = load_scene_model()
    create_folders(output_folder)  # 在输出文件夹创建分类目录
    
    # ... 其余代码保持不变，但需要将所有移动文件的目标路径改为output_folder ...