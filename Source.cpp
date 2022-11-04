////////////////////////////////Cabeceras/////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <cmath>
/////////////////////////////////////////////////////////////////////////////

///////////////////////////////Espacio de nombres////////////////////////////
using namespace cv;
using namespace std;
/////////////////////////////////////////////////////////////////////////////

/************Funcion para convertir a grises (por el metodo del promedio)***************/
Mat conv2gray(Mat imagen) {
	Mat grisesPromedio(imagen.rows, imagen.cols, CV_8UC1);//8bits Unsigned Channel 1
	double rojo, verde, azul;
	float pixelGrisProm;
	for (int i = 0;i < imagen.rows;i++) {
		for (int j = 0;j < imagen.cols;j++) {
			//obtenemos cada valor del RGB en cada pixel
			rojo = imagen.at<Vec3b>(Point(j, i)).val[2];
			verde = imagen.at<Vec3b>(Point(j, i)).val[1];
			azul = imagen.at<Vec3b>(Point(j, i)).val[0];

			pixelGrisProm = (rojo + verde + azul) / 3;//Obtencion de cada pixel de la matriz
			//Vamos guardando cada pixel en la matriz
			grisesPromedio.at<uchar>(Point(j, i)) = uchar(pixelGrisProm);
		}
	}
	return grisesPromedio;
}

/*************Funcion para creacion del kernel gaussiano****************/
Mat creacionKernel(int tamKernel, int sigma) {
	int i, j;//VARIABLES PARA RECORRER LA MATRIZ

	int limite = int(tamKernel / 2);
	//Definimos una matriz llena de 0s
	Mat kernel = Mat::zeros(tamKernel, tamKernel, CV_32F);
	i = 0;
	for (int k = -limite; k <= limite; k++) {
		j = 0;
		for (int l = -limite; l <= limite; l++) {
			//Recorremos la matriz con tamaño indicado y le vamos asigando el valor segun la formula de gauss
			kernel.at<float>(i, j) = (1 / (2 * 3.1416 * pow(sigma, 2))) * (exp(-((pow(l, 2) + pow(k, 2)) / 2 * pow(sigma, 2))));
			j++;
		}
		i++;
	}
	//cout << kernel << endl;
	return kernel;
}

/************Funcion para generar los bordes a la imagen*****************/
Mat generarBordes(Mat imgGrises, int limite) {
	//Generamos las fila y cols ampliadas, siendo que va a ser el limite *2, para generar los bordes
	int fila_amp = imgGrises.rows + limite * 2;
	int columna_amp = imgGrises.cols + limite * 2;
	//los bordes que se crean van a estar en 0s
	Mat bordesImg = Mat::zeros(fila_amp, columna_amp, CV_8UC1);
	for (int i = limite; i < fila_amp - limite; i++) {
		for (int j = limite; j < columna_amp - limite; j++) {
			bordesImg.at<uchar>(Point(i, j)) = imgGrises.at<uchar>(Point(i, j));
		}
	}
	return bordesImg;
}

/*************Funcion para aplicar un filtro****************/
Mat aplicarFiltro(Mat img, Mat kernel, float suma) {

	int limite = int(kernel.rows / 2);
	//cout << limite;
	int filas = img.rows - (limite * 2);
	int cols = img.cols - (limite * 2);
	Mat imagenFiltrada(filas, cols, CV_8UC1);
	//cout << imagenFiltrada.rows;
	//Vamos a ir recorriendo la imagen y pasando el kernel correspondiente segun sea el tamaño de este
	//vamos a ir recorriendo el kernel dentro de la imagen
	//Y aplicando la suma de productos para obtener el pixel central
	for (int ii = 0;ii < filas;ii++) {
		for (int ji = 0;ji < cols;ji++) {
			float operacion = 0.0f;
			for (int ik = 0;ik < kernel.rows;ik++) {
				for (int jk = 0;jk < kernel.cols;jk++) {
					int x = ik - limite;
					int y = -jk + limite;
					operacion = operacion + ((kernel.at<float>(ik, jk)) * (static_cast<float>(img.at<uchar>(ii + limite + x, ji + limite + y))));
				}
			}
			imagenFiltrada.at<uchar>(ii, ji) = (uchar)((int)abs(operacion/suma));
		}
	}
	return imagenFiltrada;
}

/*******Funcion para obtener la magnitud del gradiente********/
Mat magGrad(Mat fx,Mat fy) {
	Mat imgMagGrad(fx.rows, fx.cols, CV_8UC1);
	//Obtenemos una matriz con la magnitud |G|=|Gx|+|Gy|
	for (int i = 0;i < fx.rows;i++) {
		for (int j = 0;j < fx.cols;j++) {
			imgMagGrad.at<uchar>(i, j) = abs(fx.at<uchar>(i, j)) + abs(fy.at<uchar>(i, j));
		}
	}
	return imgMagGrad;
}

/*******Funcion para obtener la orientacion del gradiente********/
Mat angG(Mat fx, Mat fy) {
	Mat aux(fx.rows, fx.cols, CV_64F);
	//Creamos la matriz que nos va a dar la direccion de G
	//thetaG=arttan(Gx/Gy)
	//y lo pasamos a grados, ya que lo devuelve en radianes
	for (int i = 0;i < fx.rows;i++) {
		for (int j = 0;j < fx.cols;j++) {
			//convertimos a double, porque atan trabaja condatos de tipo double
			double gX = static_cast<float>(fx.at<uchar>(i, j));
			double gY = static_cast<float>(fy.at<uchar>(i, j));
			aux.at<double>(i, j) = (double)((atan(gY / gX))*180/3.1415);
		}
	}
	return aux;
}

/*********Funcion para sacar el Non-Max Suppression*********/
Mat NMS(Mat imgMagGrad, Mat imgAngGrad) {
	int filas = imgMagGrad.rows;
	int cols = imgMagGrad.cols;
	Mat result = Mat::zeros(filas, cols, CV_8UC1);
	float x, y, auxA, auxM;

	for (int i = 1;i < filas-1;i++) {
		for (int j = 1;j < cols-1;j++) {
			//Para operar convertimos los pixeles a float
			auxA = static_cast<float>(imgAngGrad.at<uchar>(i, j));
			//Angulo entre 0 y 23 (angulo 0)
			//Comparamos con los vecinos horizontales
			if ((0 <= auxA < 23) || (158 <= auxA <= 180)) {
				x = static_cast<float>(imgMagGrad.at<uchar>(i, j + 1));
				y = static_cast<float>(imgMagGrad.at<uchar>(i, j - 1));
			}
			//Angulo entre 23 y 68 (angulo 45)
			//Comparamos con la diagonal 
			else if (23<=auxA<68) {
				x = static_cast<float>(imgMagGrad.at<uchar>(i + 1, j - 1));
				y = static_cast<float>(imgMagGrad.at<uchar>(i - 1, j + 1));
			}
			//Angulo entre 68 y 113 (angulo 90)
			//Comparamos con la vertical
			else if (68 <= auxA < 113) {
				x = static_cast<float>(imgMagGrad.at<uchar>(i + 1, j));
				y = static_cast<float>(imgMagGrad.at<uchar>(i - 1, j));
			}
			//Angulo entre 113 y 158 (angulo 135)
			//Comparamos con la diagonal
			else if (113 <= auxA < 158) {
				x = static_cast<float>(imgMagGrad.at<uchar>(i - 1, j - 1));
				y = static_cast<float>(imgMagGrad.at<uchar>(i + 1, j + 1));
			}

			//Vemos si el valor de x,y si es mayor que los vecinos con los que se compara, se queda el pixel mayor
			//si no, se hace 0
			auxM = static_cast<float>(imgMagGrad.at<uchar>(i, j));
			if ((auxM >= x)  && (auxM >= y)) {
				result.at<uchar>(i, j) = imgMagGrad.at<uchar>(i, j);
			}
			else {
				result.at<uchar>(i, j) =0;
			}

		}
	}
	return result;
}

/**********Funcion para hacer el doble umbralizado, con valores propuestos*************/
Mat umbralizado(Mat img, int umbralLow, int umbralHigh) {
	Mat result(img.rows, img.cols, CV_8UC1);
	//Con los umbrales propuestos vemos,
	//si el valor del pixel es mayor que umbralHigh se mantiene (pixel fuerte)
	//Si esta entre el umbralLow y umbralHigh se le asigna 25 (pixel debil)
	//Si es menor que umbralLow se asigna a 0 ya que se discrimina (pixel no relevante)
	for (int i = 0;i < img.rows;i++) {
		for (int j = 0;j < img.cols;j++) {
			if (img.at<uchar>(i, j) >= umbralHigh) {
				result.at<uchar>(i, j) = 255;
			}
			else if ((umbralLow < img.at<uchar>(i, j)) and (img.at<uchar>(i,j)>=umbralHigh)) {
				result.at<uchar>(i, j) =25;
			}
			else {
				result.at<uchar>(i, j) = 0;
			}
		}
	}
	return result;
}

/***********Funcion para la histeresis con umbrales propuestos*************/
Mat hysteresis(Mat img,int debil, int fuerte) {
	//si el valor de un pixel es igual al pixel debi, vemos si alguno de sus vecinos es igual al pixel fuerte, y si lo es asignamos el pixel fuerte
	//si ningun vecino es igual a pixel fuerte se asgna a 0
	float aux;
	for (int i = 1;i < img.rows - 1;i++) {
		for (int j = 0;j < img.cols - 1;j++) {
			aux = img.at<uchar>(i, j);
			if (aux == debil) {
				if ((img.at<uchar>(i + 1, j - 1) == fuerte) or (img.at<uchar>(i + 1, j) == fuerte) or (img.at<uchar>(i + 1, j + 1) == fuerte) or (img.at<uchar>(i, j - 1) == fuerte) or (img.at<uchar>(i, j + 1) == fuerte) or (img.at<uchar>(i - 1, j - 1) == fuerte) or (img.at<uchar>(i - 1, j) == fuerte) or (img.at<uchar>(i - 1, j + 1) == fuerte)) {
					img.at<uchar>(i, j) = fuerte;
				}
				else {
					img.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return img;
}

/////////////////////////Inicio de la funcion principal///////////////////
int main() {


	/********Declaracion de variables generales*********/
	char NombreImagen[] = "lena.png";
	Mat imagen; // Matriz que contiene nuestra imagen sin importar el formato
	/************************/

	/*********Lectura de la imagen*********/
	imagen = imread(NombreImagen);

	if (!imagen.data) {
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}
	/*******CONVERSION A GS*********/
	Mat grisesFun = conv2gray(imagen);

	/******CREACION DEL KERNEL******/
	int tamKernel, sigma;
	//Pedimos al usuario el tamaño del kernel, y valor de sigma
	cout << "Introduzca el tamano del kernel:" << endl;
	cin >> tamKernel;
	cout << "Introduzca el valor de sigma:" << endl;
	cin >> sigma;

	Mat kernelResult=creacionKernel(tamKernel,sigma);
	cout << kernelResult << endl;
	/********Normalizado******/
	float suma = 0.0f;
	for (int i = 0;i < kernelResult.rows;i++) {
		for (int j = 0;j < kernelResult.cols;j++) {
			suma = suma + kernelResult.at<float>(i, j);
		}
	}

	int limite = int(tamKernel / 2);//Da el numero entero de filas y columnas a agregar

	/************GENERACION DE BORDES*************/
	Mat bordesImg = generarBordes(grisesFun, limite);

	/*****APLICACION DE FILTRO GAUSSIANO******/
	Mat imagenFiltrada = aplicarFiltro(bordesImg, kernelResult, suma);

	/********Ecualizacion********/
	Mat imgEcualizada;
	cv::equalizeHist(imagenFiltrada, imgEcualizada);
	

	/********APLICACION DE SOBEL**********/
	//Generamos las dos  matrices para Gx y Gy
	Mat gx = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat gy = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	//Primero le generamos bordes
	Mat bordesSobel = generarBordes(imagenFiltrada, 1);
	//Aplicamos gx y gy a la imagenFiltrada
	Mat imgFx = aplicarFiltro(bordesSobel, gx,1.0);
	Mat imgFy = aplicarFiltro(bordesSobel, gy,1.0);

	/************************/
	//Obtenemos |G| la magnitud del gradiente
	Mat imgMagGrad = magGrad(imgFx, imgFy);
	//Angulo de G, orientacion del gradiente
	Mat imgAngG(imgFx.rows, imgFy.cols, CV_8UC1);
	Mat aux = angG(imgFx, imgFy);
	aux.convertTo(imgAngG, CV_8U);//Regresamos a uchar
	/***********************/
	//Non-Max_suppresion
	Mat imgNMS = NMS(imgMagGrad, imgAngG);

	//Doble umbralizado
	Mat imgUmbral = umbralizado(imgNMS, 14, 75);

	//Hysteresis
	Mat imgHyster = hysteresis(imgUmbral, 25, 255);
	

	//Impresion de tamaños de imagen
	cout << "tamano original: " << imagen.rows << "x" << imagen.cols << endl;
	cout << "tamano original grises: " << grisesFun.rows << "x" << grisesFun.cols << endl;
	cout << "tamano con bordes: " << bordesImg.rows << "x" << bordesImg.cols << endl;
	cout << "tamano filtrada: " << imagenFiltrada.rows << "x" << imagenFiltrada.cols << endl;
	cout << "tamano ecualizada: " << imgEcualizada.rows << "x" << imgEcualizada.cols << endl;
	cout << "tamano bordes sobel: " << bordesSobel.rows << "x" << bordesSobel.cols << endl;
	cout << "tamano fx: " << imgFx.rows << "x" << imgFx.cols << endl;
	cout << "tamano fy: " << imgFy.rows << "x" << imgFy.cols << endl;
	cout << "tamano |G|: " << imgMagGrad.rows << "x" << imgMagGrad.cols << endl;
	cout << "tamano anguloG: " << imgAngG.rows << "x" << imgAngG.cols << endl;
	cout << "tamano NMS: " << imgNMS.rows << "x" << imgNMS.cols << endl;
	cout << "tamano umbral: " << imgUmbral.rows << "x" << imgUmbral.cols << endl;
	cout << "tamano Hyster: " << imgHyster.rows << "x" << imgHyster.cols << endl;

	//Mostramos las imagenes
	imshow("Img original", imagen);
	imshow("Img grises", grisesFun);
	imshow("Img con Bordes", bordesImg);
	imshow("Img filtrada", imagenFiltrada);	
	imshow("Img ecualizada", imgEcualizada);
	imshow("Img Fx", imgFx);
	imshow("Img Fy", imgFy);
	imshow("Img |G|", imgMagGrad);
	imshow("Img angulo G", imgAngG);
	imshow("Img NMS", imgNMS);
	imshow("Img umbral", imgUmbral);
	imshow("Img Hyster", imgHyster);

	waitKey(0); //Función para esperar
	return 1;
}

/////////////////////////////////////////////////////////////////////////