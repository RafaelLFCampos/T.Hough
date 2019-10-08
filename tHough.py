import cv2
import numpy as np
import math

#Hough Linhas
nomeImagem = 'damas.png'
img = cv2.imread(nomeImagem)
#mesc utilizado para fazer exibir as linhas e circulos detectados na imagem
mesc = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 200)
#aplicando a transformada de hough para detectar linhas
linhas = cv2.HoughLines(canny, 1, math.pi/180.0, 100, np.array([]), 0, 0)

a,b,c = linhas.shape
for i in range(a):
    rho = linhas[i][0][0]
    theta = linhas[i][0][1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0, y0 = a*rho, b*rho
    pt1 = (int(x0+1000*(-b)), int(y0+1000*(a)))
    pt2 = (int(x0-1000*(-b)), int(y0-1000*(a)))
    cv2.line(img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(mesc, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

#Hough Círculos
imagem = cv2.imread(nomeImagem)
saida = imagem.copy()
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
#aplicando a transformada de hough para detectar círculos
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=65, param2=50, minRadius=0, maxRadius=0)
detected_circles = np.uint16(np.around(circles))

for (x, y ,r) in detected_circles[0, :]:
    cv2.circle(saida, (x, y), r, (0, 255, 255), 2)
    cv2.circle(saida, (x, y), 2, (0, 0, 255), 2)
    cv2.circle(mesc, (x, y), r, (0, 255, 255), 2)
    cv2.circle(mesc, (x, y), 2, (0, 0, 255), 2)
   
cv2.imshow("Imagem Original / H. Linhas / H. Circulos / H. Linhas + H. Circulos", np.hstack([imagem, img, saida, mesc]))
cv2.imwrite("h_linhas.png", img)
cv2.imwrite("h_circulos.png", saida)
cv2.imwrite("h_linhasEcirculos.png", mesc)
cv2.waitKey(0)
cv2.destroyAllWindows()
