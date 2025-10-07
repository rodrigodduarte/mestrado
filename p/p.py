from PIL import Image

# caminhos das imagens (ajuste para os arquivos reais)
img1_path = "/home/rodrigoduarte/Documentos/mestrado/p/l1nr005.png"
img2_path = "/home/rodrigoduarte/Documentos/mestrado/p/R_90Class1 (1).jpg"
img3_path = "/home/rodrigoduarte/Documentos/mestrado/p/Class13(1)R0_00007.jpg"

# abrir imagens
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)
img3 = Image.open(img3_path)

# padronizar altura para todas ficarem iguais
base_height = 300
def resize_to_height(img, height):
    w, h = img.size
    new_w = int(w * (height / h))
    return img.resize((new_w, height))

img1 = resize_to_height(img1, base_height)
img2 = resize_to_height(img2, base_height)
img3 = resize_to_height(img3, base_height)

# calcular dimensões da imagem final
total_width = img1.width + img2.width + img3.width
max_height = base_height

# criar tela em branco
new_img = Image.new("RGB", (total_width, max_height), (255, 255, 255))

# colar as imagens na ordem: swedish, flavia, d2
x_offset = 0
for im in [img1, img2, img3]:
    new_img.paste(im, (x_offset, 0))
    x_offset += im.width

# salvar resultado
new_img.save("datasets.png")
print("Imagem final salva como datasets.png")
