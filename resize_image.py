import cv2

style = cv2.imread("./style.jpg")
content = cv2.imread("./skyline.jpg")

print(f"Style size {style.shape}")
print(f"Content size {content.shape}")

h, w, c = style.shape
content2 = cv2.resize(content, (w, h))
print(f"Content2 size {content2.shape}")
cv2.imwrite("./skyline2.jpg", content2)
