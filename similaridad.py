import os
import networkx as nx
# from copy import copy
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2
import math
from PIL import Image
# import glob
import multiprocessing
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
# import xml.etree.ElementTree as ET
import motmetrics as mm
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.metrics import f1_score, mean_squared_error
from shapely.geometry import Point, Polygon
from quadtree import QuadTree
from quadtree import Point as QPoint
from quadtree import Rect as QRect
from itertools import combinations
from itertools import islice


def init_escena(path_video, diferencia=5):
    video = iio.imopen(path_video, io_mode="r", plugin="pyav")
    altura, largo, _ = video.read(index=0).shape
    coordenadas = [(diferencia, diferencia),
                   (diferencia, altura - diferencia),
                   (largo - diferencia, altura - diferencia),
                   (largo - diferencia, diferencia)]
    escena = Polygon(coordenadas)
    fps = video.metadata()['fps']
    return video, escena, largo, altura, fps


def convertir_entero(elemento):
    # Elemento tiene este formato: 'uadetrac11_1.txt' 'uadetrac11_10.txt'
    a = elemento[:-4]
    b = a.split("_")
    c = int(b[-1])
    return c


def leer_archivos(path_labels):
    archivos = os.listdir(path_labels)
    archivos_ordenados = sorted(archivos, key=convertir_entero)
    for elemento in archivos_ordenados:
        yield os.path.join(path_labels, elemento)


def leer_frame(video, frame_number):
    frame = video.read(index=frame_number)
    return frame


def leer_xys(archivo):
    with open(archivo, 'r') as archivo_video:
        lineas = archivo_video.readlines()
        for linea in lineas:
            datos = linea.strip()
            _, *puntos, conf = datos.split()
            puntos = [int(p) for p in puntos]
            final = puntos + [float(conf)]
            yield tuple(final)


def obtener_objetos(archivo, frame):
    xys = leer_xys(archivo)  # xys es el resultado de leer archivo
    for xi, yi, xd, yd, c in xys:
        #    print(f"{xd - xi} {yd - yi}")
        obj = frame[yi:yd, xi:xd, :]  # Extraer la región correspondiente al objeto
        # plt.imshow(obj)
        # plt.show()
        yield obj


class Nodo:
    def __init__(self, xi, yi, xd, yd, c):
        self.id = 0
        self.xi = xi
        self.yi = yi
        self.xd = xd
        self.yd = yd
        self.c = c

    def centro(self):
        centro_x = self.xi + (self.xd - self.xi) / 2
        centro_y = self.yi + (self.yd - self.yi) / 2
        return centro_x, centro_y

    def suma(self, numero):
        self.id = 0
        self.xi += numero
        self.yi += numero
        self.xd += numero
        self.yd += numero

    def __le__(self, otro):
        return (self.xi <= otro.xi and
                self.yi <= otro.yi and
                self.xd <= otro.xd and
                self.yd <= otro.yd and
                self.c <= otro.c)

    def __lt__(self, otro):
        return (self.xi <= otro.xi and
                self.yi < otro.yi and
                self.xd < otro.xd and
                self.yd < otro.yd and
                self.c < otro.c)

    def __repr__(self):
        nombre = self.__class__.__name__
        return f"{nombre} {self.id} {self.centro()} {self.c}"


def comparar_nodos(izq, der):
    return (izq.xi == der.xi and
            izq.yi == der.yi and
            izq.xd == der.xd and
            izq.yd == der.yd and
            math.isclose(izq.c, der.c, abs_tol=1e-6))


def crear_nodos(archivo):
    points = list(leer_xys(archivo))
    unique_points = []

    # Set a threshold for similarity
    threshold = 2.0

    # Loop through each point in the original list
    for identificador, point in enumerate(points):
        is_unique = True
        # Calculate similarity or distance with points already in the unique list
        for unique_point in unique_points:
            distance = calcular_distancia_centro(Nodo(*point), unique_point)
            if distance < threshold and calcular_iou(Nodo(*point), unique_point) > 0.7:
                is_unique = False
                break
        # If the point is unique, add it to the unique list
        if is_unique:
            nodo = Nodo(*point)
            nodo.id = identificador
            unique_points.append(nodo)

    # print("Original points:", points)
    # print("Unique points:", unique_points)
    return unique_points


def calcular_iou(na, nb):
    # Coordenadas del rectangulo de interseccion
    xa = max(na.xi, nb.xi)
    ya = max(na.yi, nb.yi)
    xb = min(na.xd, nb.xd)
    yb = min(na.yd, nb.yd)
    inter_area = max(0, xb - xa) * max(0, yb - ya)

    area_na = abs(na.xi - na.xd) * abs(na.yi - na.yd)
    area_nb = abs(nb.xi - nb.xd) * abs(nb.yi - nb.yd)

    iou = inter_area / (area_na + area_nb - inter_area)

    return iou


def calcular_similaridad_estructural(obj1, obj2, min_size=7):
    # Ensure minimum size
    a, b = max(obj1.shape[1], min_size), max(obj1.shape[0], min_size)
    c, d = max(obj2.shape[1], min_size), max(obj2.shape[0], min_size)

    # Resize images
    obj1 = cv2.resize(obj1, (a, b))
    obj2 = cv2.resize(obj2, (c, d))

    # Resize the larger image to match dimensions
    ratio = (a * b) / (c * d)
    # if ratio > 2.5:
    #    return 0
    if ratio > 1:
        obj2 = cv2.resize(obj2, (a, b))
    else:
        obj1 = cv2.resize(obj1, (c, d))
    s = ssim(obj1, obj2, channel_axis=2, data_range=255)

    return s


def calcular_similaridad_estructural_prev(obj1, obj2, min_size=7):
    a, b = max(obj1.shape[1], min_size), max(obj1.shape[0], min_size)
    c, d = max(obj2.shape[1], min_size), max(obj2.shape[0], min_size)

    if (a, b) != (c, d):
        obj1 = cv2.resize(obj1, (c, d))
        obj2 = cv2.resize(obj2, (c, d))

    s = ssim(obj1, obj2, channel_axis=2)
    return s


def calcular_similaridad_estructural_40(obj1, obj2, min_size=40):
    a, b = max(obj1.shape[1], min_size), max(obj1.shape[0], min_size)
    c, d = max(obj2.shape[1], min_size), max(obj2.shape[0], min_size)

    if (a, b) <= (c, d):
        obj1 = cv2.resize(obj1, (c, d))
        obj2 = cv2.resize(obj2, (c, d))
    else:
        obj1 = cv2.resize(obj1, (a, b))
        obj2 = cv2.resize(obj2, (a, b))

    s = ssim(obj1, obj2, channel_axis=2)
    return s


def obtener_posibles_aristas_similaridad(lista_nodos_izq, lista_obj_izq,
                                         lista_nodos_der, lista_obj_der, dif_frames):
    for izquierdo in lista_nodos_izq:
        izquierdo.id = f"{izquierdo.id} izq"
    for derecho in lista_nodos_der:
        derecho.id = f"{derecho.id} der"

    lista_de_listas = []
    for izquierdo, obj_izquierdo in zip(lista_nodos_izq, lista_obj_izq):
        lista_temporal = []
        for derecho, obj_derecho in zip(lista_nodos_der, lista_obj_der):
            dist_centro = 50
            iou_score = calcular_iou(izquierdo, derecho)
            centro = calcular_distancia_centro(izquierdo, derecho)
            if dif_frames > 1:
                dist_centro *= dif_frames
            if (iou_score >= 0.3 and centro <= dist_centro):
                similaridad = calcular_similaridad_estructural(obj_izquierdo, obj_derecho)
                if similaridad >= 0.3:
                    tupla = (izquierdo, derecho, similaridad, iou_score, centro)
                    lista_temporal.append(tupla)
        lista_de_listas.extend(lista_temporal)

    return lista_de_listas


def numeracion(nombre_archivo):
    # nombre_archivo = "uadetrac11_1.txt"
    primera_parte = nombre_archivo.split(".")[0]
    numero = int(primera_parte.split("_")[-1])
    return numero


def comparar_frames_similaridad(path_labels, path_video):
    generador_archivos = leer_archivos(path_labels)
    archivo1 = next(generador_archivos)
    numero = numeracion(archivo1)
    frame1 = leer_frame(path_video, numero - 1)
    
    for archivo2 in generador_archivos:
        siguiente = numeracion(archivo2)
        frame2 = leer_frame(path_video, siguiente - 1)

        lista_nodos_izq = crear_nodos(archivo1)
        lista_nodos_der = crear_nodos(archivo2)

        lista_obj_izq = obtener_objetos(archivo1, frame1)
        gen_lista_obj_izq = list(lista_obj_izq)

        lista_obj_der = obtener_objetos(archivo2, frame2)
        gen_lista_obj_der = list(lista_obj_der)

        # dif_nodos
        diff = siguiente - numero
        val_similaridad = obtener_posibles_aristas_similaridad(
            lista_nodos_izq, gen_lista_obj_izq,
            lista_nodos_der, gen_lista_obj_der, diff)
        if diff == 1:
            yield val_similaridad
        else:
            for k in range(diff):
                yield val_similaridad
        archivo1 = archivo2
        frame1 = frame2
        numero = siguiente


def posibles_pesos(path_labels, path_video):
    a = 0.4
    b = 0.6
    lista_de_listas = []
    for elementos in comparar_frames_similaridad(path_labels, path_video):
        lista_temporal = []
        for elemento in elementos:
            nodo_izquierdo = elemento[0]
            nodo_derecho = elemento[1]
            peso_similaridad = elemento[2]
            peso_iou = elemento[3]
            if peso_iou == 0:
                a = 1
            peso_final = a * peso_similaridad + b * peso_iou
            tupla = (nodo_izquierdo, nodo_derecho, peso_final)
            lista_temporal.append(tupla)
        lista_de_listas.append(lista_temporal)

    return lista_de_listas


def crear_grafo_bipartido_pesos(path_labels, path_video):
    lista_grafos = []
    generador_posibles_aristas = posibles_pesos(path_labels, path_video)
    for posibles_aristas in generador_posibles_aristas:
        # print(f'Grafo {i}')
        nodos1 = set()
        nodos2 = set()
        lista_aristas = []
        for a, b, w in posibles_aristas:
            nodos1.add(a)
            nodos2.add(b)
            lista_aristas.append((a, b, -w))

        grafo = nx.Graph()
        grafo.add_nodes_from(nodos1, bipartite=0)
        grafo.add_nodes_from(nodos2, bipartite=1)

        grafo.add_weighted_edges_from(lista_aristas)
        lista_grafos.append(grafo)
    return lista_grafos


def obtener_matching_pesos(path_labels, path_video):
    lista_matchings = []
    grafos = crear_grafo_bipartido_pesos(path_labels, path_video)
    for grafo in grafos:
        matching = nx.min_weight_matching(grafo, weight="weight")
        matching_ordenado = []
        for nod_izq, nod_der in matching:
            if nod_izq.id.endswith('der'):
                matching_ordenado.append((nod_der, nod_izq))
            else:
                matching_ordenado.append((nod_izq, nod_der))

        grafo_matching = nx.Graph()
        # pesos = posibles_pesos(path_labels, path_video)
        # weight = pesos[2]
        # >> Comprobar esto
        grafo_matching.add_edges_from((a, b, {"weight": calcular_iou(a, b)}) for a, b in matching_ordenado)
        lista_matchings.append(grafo_matching)

    return lista_matchings


def calcular_distancia_centro(objeto1, objeto2):
    x1, y1 = objeto1.centro()
    x2, y2 = objeto2.centro()
    distancia = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distancia


def obtener_objetos_en_frame(frame_deseado, correspondencias_parado):
    resultados_frame = {}

    for clave, valores in correspondencias_parado.items():
        for frame, nodo in valores:
            if frame == frame_deseado:
                if clave in resultados_frame:
                    resultados_frame[clave].append((frame, nodo))
                else:
                    resultados_frame[clave] = [(frame, nodo)]

    return resultados_frame


def buscar_puntos_cercanos(correspondencias, ids_anomalias, nodo, dimensions, radius=50):
    WIDTH, HEIGHT = dimensions
    domain = QRect(WIDTH/2, HEIGHT/2, WIDTH, HEIGHT)
    qtree = QuadTree(domain)

    check_ids = ids_anomalias.union(islice(reversed(correspondencias.keys()), 50))
    for key in check_ids:
        lista_nodos = correspondencias[key]
        _, ult_nodo = lista_nodos[-1]
        point = QPoint(*ult_nodo.centro(), key)
        qtree.insert(point)

    found_points = []
    qtree.query_radius(nodo.centro(), radius, found_points)
    points = [p.payload for p in found_points]
    return points


def rastreamento_vehiculo_parado(path_labels, path_video, val_similaridad=0.3):
    VIDEO, ESCENA, WIDTH, HEIGHT, FPS = init_escena(path_video, 10)
    dimensions = WIDTH, HEIGHT
    LONGITUD_POSIBLES = 150
    correspondencias = obtener_matching_pesos(path_labels, VIDEO)
    posibles_anomalias = set()
    anomalias = set()
    correspondencias_parado = {}
    vehiculos_en_poligono = {}
    poligono = None
    total_frames = len(correspondencias) + 1

    grafo = correspondencias[0]
    for j, (nodo_a, nodo_b) in enumerate(grafo.edges, 1):
        if not ESCENA.contains(Point(nodo_a.centro())):
            continue
        correspondencias_parado[f'obj{j}'] = [(1, nodo_a), (2, nodo_b)]

    for frame in range(1, total_frames - 1):
        grafo = correspondencias[frame]
        longitud_base = len(correspondencias_parado)

        for k, (nodo_a, nodo_b) in enumerate(grafo.edges):
            if not ESCENA.contains(Point(nodo_a.centro())):
                continue
            if poligono:
                siguiente_elemento_inf = Point(nodo_a.xi, nodo_a.yi)
                siguiente_elemento_sup = Point(nodo_a.xd, nodo_a.yd)
                centro_siguiente_elemento = Point(nodo_a.centro())
                if not (poligono.contains(centro_siguiente_elemento) or poligono.contains(siguiente_elemento_inf)
                        or poligono.contains(siguiente_elemento_sup)):
                    continue
            agregado = False
            points = buscar_puntos_cercanos(correspondencias_parado, posibles_anomalias, nodo_a, dimensions)
            # for id_obj, lista_nodos in reversed(correspondencias_parado.items()):
            for id_obj in points:
                lista_nodos = correspondencias_parado[id_obj]
                if poligono and len(lista_nodos) > LONGITUD_POSIBLES:
                    posibles_anomalias.add(id_obj)
                _, ultimo_nodo = lista_nodos[-1]
                if comparar_nodos(ultimo_nodo, nodo_a):
                    correspondencias_parado[id_obj].append((frame + 2, nodo_b))
                    agregado = True
                    break

            if not agregado:
                agregado_por_similaridad = False

                yi, yd = nodo_a.yi, nodo_a.yd
                xi, xd = nodo_a.xi, nodo_a.xd
                actual = leer_frame(VIDEO, frame + 1)
                a_objeto = actual[yi:yd, xi:xd, :]

                if len(posibles_anomalias) > 1:
                    for id_izq, id_der in combinations(posibles_anomalias, 2):
                        f, n_izq = correspondencias_parado[id_izq][-1]
                        f, n_der = correspondencias_parado[id_der][0]
                        centro = calcular_distancia_centro(n_izq, n_der)
                        iou = calcular_iou(n_izq, n_der)
                        # calcular_similaridad_estructural()
                        if centro < 25 and iou > 0.5:
                            anomalias.add(id_izq)
                            anomalias.add(id_der)
                # points = buscar_puntos_cercanos(correspondencias_parado, nodo_a)
                ids_similares = []
                for id_obj in points:
                    lista_nodos = correspondencias_parado[id_obj]
                    if len(lista_nodos) < 20:
                        continue
                    n_ult_frame, ultimo_nodo = lista_nodos[-1]
                    n_pri_frame, primer_nodo = lista_nodos[0]
                    if len(lista_nodos) >= 50:
                        n_pri_frame, primer_nodo = lista_nodos[-50]
                    
                    if abs(frame - n_ult_frame) > 50:
                        continue

                    if (calcular_iou(nodo_a, ultimo_nodo) <= 0.5
                        or calcular_distancia_centro(nodo_a, ultimo_nodo) >= 25):
                        continue
                        # and calcular_iou(nodo_a, primer_nodo) <= 0.5

                    distancia_x = abs(ultimo_nodo.centro()[0] - primer_nodo.centro()[0])
                    distancia_y = abs(ultimo_nodo.centro()[1] - primer_nodo.centro()[1])
                    tiempo_total = n_ult_frame - n_pri_frame
                    velocidad_x = distancia_x / (tiempo_total * FPS)
                    velocidad_y = distancia_y / (tiempo_total * FPS)
                    
                    pos_x_init, pos_y_init = ultimo_nodo.centro()
                    pos_x = velocidad_x * (frame - n_ult_frame)
                    pos_y = velocidad_y * (frame - n_ult_frame)
                    pos_x_final = pos_x_init + pos_x
                    pos_y_final = pos_y_init + pos_y
                    na_x, na_y = nodo_a.centro()
                    
                    if abs(na_x - pos_x_final) > 10 and abs(na_y - pos_y_final) > 10:
                        continue

                    yi, yd = ultimo_nodo.yi, ultimo_nodo.yd
                    xi, xd = ultimo_nodo.xi, ultimo_nodo.xd
                    ultimo_frame = leer_frame(VIDEO, n_ult_frame)
                    ultimo_objeto = ultimo_frame[yi:yd, xi:xd, :]
                    valor_similar_calc = calcular_similaridad_estructural(ultimo_objeto, a_objeto)

                    if valor_similar_calc >= val_similaridad:
                        ids_similares.append((valor_similar_calc, id_obj))

                # Obtener la similaridad y escoger la mayor
                if ids_similares:
                    # Todo: Elegir el objeto con más tiempo en el video
                    _, id_obj = max(ids_similares)
                    correspondencias_parado[id_obj].append((frame + 2, nodo_b))
                    agregado_por_similaridad = True

                if not agregado_por_similaridad:
                    id_obj = f'obj{longitud_base + 1}'
                    correspondencias_parado[id_obj] = [(frame + 1, nodo_a), (frame + 2, nodo_b)]

        if frame == 9000:
            coordenadas_movimiento = []
            for id_obj, lista_nodos in correspondencias_parado.items():
                vehiculos = []
                for k in range(1, len(lista_nodos)):
                    if calcular_iou(lista_nodos[k - 1][1], lista_nodos[k][1]) <= 0.9 \
                            or calcular_distancia_centro(lista_nodos[k - 1][1], lista_nodos[k][1]) >= 1:
                        vehiculos.append(lista_nodos[k - 1])

                if len(vehiculos) >= round(1 / 4 * len(lista_nodos)) \
                        or calcular_distancia_centro(lista_nodos[0][1], lista_nodos[-1][1]) >= 40:
                    vehiculos_en_poligono[id_obj] = lista_nodos

                    nodo_inicial = vehiculos_en_poligono[id_obj][0][1]
                    inicial1 = nodo_inicial.xi, nodo_inicial.yi
                    inicial2 = nodo_inicial.xd, nodo_inicial.yd
                    inicial3 = nodo_inicial.xd, nodo_inicial.yi
                    inicial4 = nodo_inicial.xi, nodo_inicial.yd

                    nodo_final = vehiculos_en_poligono[id_obj][-1][1]
                    final1 = nodo_final.xi, nodo_final.yi
                    final2 = nodo_final.xd, nodo_final.yd
                    final3 = nodo_final.xd, nodo_final.yi
                    final4 = nodo_final.xi, nodo_final.yd
                    coordenadas_movimiento.extend(
                        [inicial1, inicial2, inicial3, inicial4, final1, final2, final3, final4])

            coordenadas = np.array(coordenadas_movimiento)
            hull = ConvexHull(coordenadas)
            vertices_indices = hull.vertices
            vertices_coordenadas = coordenadas[vertices_indices]
            poligono = Polygon(vertices_coordenadas)

            diferencia_keys = set(correspondencias_parado.keys()) - set(vehiculos_en_poligono.keys())
            diferencia_keys = list(diferencia_keys)
            diferencia_keys.sort(key=lambda x: int(x[3:]))

            for key in diferencia_keys:
                lista_nodos = correspondencias_parado[key]
                inferior = lista_nodos[0][1].xi, lista_nodos[0][1].yi
                punto1 = Point(inferior)
                superior = lista_nodos[0][1].xd, lista_nodos[0][1].yd
                punto2 = Point(superior)
                if poligono.contains(punto1) and poligono.contains(punto2):
                    vehiculos_en_poligono[key] = lista_nodos

            # Reorganizar el diccionario para que los nombres de los objetos comiencen desde 'obj1'
            contador_objetos = 1
            nuevo_vehiculos_en_poligono = {}
            for id_obj, lista_nodos in vehiculos_en_poligono.items():
                nuevo_id_obj = f'obj{contador_objetos}'
                nuevo_vehiculos_en_poligono[nuevo_id_obj] = lista_nodos
                contador_objetos += 1

            # Reemplazar el diccionario original con el diccionario reorganizado
            correspondencias_parado = nuevo_vehiculos_en_poligono
            for k, ln in correspondencias_parado.items():
                if len(ln) > LONGITUD_POSIBLES:
                    posibles_anomalias.add(k)
    # print(len(correspondencias_parado))
    # print(anomalias)
    return correspondencias_parado, anomalias


def procesar_anomalias(correspondencias, anomalias):
    checking = {}
    if not anomalias:
        return checking
    vistos = set()
    # anomalias = {'obj231', 'obj269', 'obj6', 'obj99', 'obj113'}
    anomalias = sorted(anomalias, key=lambda x: int(x[3:]))
    ida = 1
    for ka in anomalias:
        if ka in vistos:
            continue
        ka_unico = True 
        for kp in anomalias:
            if ka == kp or kp in vistos:
                continue
            fpri, npri = correspondencias[ka][-5]
            fsec, nsec = correspondencias[kp][5]
            centro = calcular_distancia_centro(npri, nsec)
            iou = calcular_iou(npri, nsec)
            if centro < 25 and iou > 0.5:
                presente = checking.get(ida, False)
                if not presente:
                    checking[ida] = [ka]
                    checking[ida].append(kp)
                else:
                    checking[ida].append(kp)
                vistos.add(kp)  
                ka_unico = False 
        if ka_unico:
            checking[ida] = [ka]

        vistos.add(ka)
        ida = ida + 1

    anomalias_finales = {}
    for clave, valor in checking.items():
        primer_obj = valor[0]
        primer_frame = int(correspondencias[primer_obj][0][0])
        ultimo_obj = valor[-1]
        ultimo_frame = int(correspondencias[ultimo_obj][-1][0])
        duracion = ultimo_frame - primer_frame
        if duracion > 1800:
            anomalias_finales[clave] = valor


    resultado_anomalias = []
    for clave, ids_objs in anomalias_finales.items():
        # print(clave, ids_objs)
        ids_objs = sorted(ids_objs, key=lambda x: int(x[3:]))
        # print(f"{clave}:")
        inicio = encontrar_anomalia(correspondencias, ids_objs)
        final = encontrar_anomalia(correspondencias, ids_objs[::-1], "final")
        if inicio > final:
            ultimo_id = ids_objs[-1]
            final, _ = correspondencias[ultimo_id][-1]
        # print(f"{inicio} a {final}: {'anomalia' if final - inicio >= 1800 else ''}")
        if final - inicio >= 1800:
            resultado_anomalias.append((inicio, final))
            # resultado_anomalias.append((clave, inicio, final, 'anomalia' if final - inicio >= 1800 else ''))

    return resultado_anomalias


def encontrar_anomalia(correspondencias, ids_objs, direccion="inicial"):
    K = 20
    n_anomalia = 0
    encontro = False
    for id_obj in ids_objs:
        # print(id_obj)
        nodos = correspondencias[id_obj][::K]
        n_nodos = len(nodos)
        actual = []
        n_veces = 0
        for i in range(n_nodos - 1):
            # print(nodos[i], nodos[i+1])

            frame1, nodo1 = nodos[i]
            frame2, nodo2 = nodos[i+1]
            x1, y1 = nodo1.centro()
            x2, y2 = nodo2.centro()
            distancia = ((x1 - x2)**2 + (y1 -y2)**2)**0.5
            tiempo_total = frame2 - frame1
            velocidad = distancia / tiempo_total
            # print(velocidad)

            if comparar_direccion(velocidad, direccion):
                n_veces += 1
                actual.append((i, frame1))
                if n_veces == 3 and (actual[-1][0] - actual[0][0] == 2):
                    # print(f"{direccion} {actual[0][1]}")
                    n_anomalia = actual[0][1]
                    encontro = True
                    break
            else:
                n_veces = 0
                actual = []
        if encontro:
            break
    return n_anomalia


def comparar_direccion(velocidad, direccion):
    if direccion == "inicial":
        return velocidad <= 0.1
    else:
        return velocidad >= 0.1


def etiquetas_rastreamento_video(path_video, rastreo_vehiculo):
    cap = cv2.VideoCapture(path_video)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second {fps}")
    # Crear un objeto VideoWriter para guardar el video con las etiquetas
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('Anomaly41.mp4', fourcc, fps, (video_width, video_height))

    frame_actual = 1
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        for label, nodos in rastreo_vehiculo.items():
            for num_frame, nodo in nodos:
                x, y = nodo.xi, nodo.yi  # Coordenadas del objeto

                if frame_actual == num_frame:
                    # print(label, num_frame, nodo, type(nodo.xi))
                    cv2.rectangle(frame, (int(x), int(y)),
                                  (int(x + round(nodo.xd) - round(x)), int(y + round(nodo.yd) - round(y))),
                                  (0, 0, 255), 1)  # Dibujar cuadro delimitador
                    cv2.rectangle(frame, (int(x), int(y - 20)), (int(x + 50), int(y - 10)),
                                  (0, 239, 204), -1, cv2.LINE_AA)
                    cv2.putText(frame, label, (int(x), int(y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0),
                                1)  # Escribir etiqueta
        cv2.putText(frame, f"Frame: {frame_actual - 1}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1)  # Escribir etiqueta

        frame_actual = frame_actual + 1
        output_video.write(frame)  # Guardar el frame con las etiquetas en el video de salida

    # Liberar recursos
    cap.release()
    output_video.release()

    return output_video


# Para un conjunto de videos
def procesar_video(params):
    path_labels = params["label"]
    path_video = params["video"]
    rastreo, anomalias_ids = rastreamento_vehiculo_parado(path_labels, path_video, 0.3)
    intervalos = procesar_anomalias(rastreo, anomalias_ids)
    n_video = path_labels.split('/')[-1]
    return n_video, intervalos


# def process_videos():
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    directorio_test = '/home/usp/Documents/Ana/Track960'

    dir_datos = os.listdir(directorio_test)
    dir_datos.sort(key=lambda n: int(n[9:]))

    parametros = []
    for dir_video in dir_datos:
        path_video_test = os.path.join(directorio_test, dir_video)
        sub_directories = os.listdir(path_video_test)
        a, b = sub_directories
        if not b.endswith('mp4'):
            a, b = b, a
        path_labels = os.path.join(path_video_test, a, b[:-4])
        path_video = os.path.join(path_video_test, b)


        params = {"label": path_labels,
                    "video": path_video}
        parametros.append(params)

    # labels_path = '/home/usp/Documents/Ana/Track960/detection11/labels/45'
    # video_path = '/home/usp/Documents/Ana/Track960/detection11/45.mp4'
    # parametros = []
    # params = {"label": labels_path,
    #           "video": video_path}
    # parametros.append(params)
    # parametros.append(params)

    cpus = 8
    with multiprocessing.Pool(processes=cpus) as pool:
        resultados = pool.imap_unordered(procesar_video, parametros)
        pool.close()
        pool.join()

    nombre_archivo = "datos.txt"

    resultados = list(resultados)
    resultados.sort()

    with open(nombre_archivo, 'w') as archivo:
        for tupla in resultados:
            video_num = "Video" + str(tupla[0])
            valores = " ".join(map(str, [x for t in tupla[1] for x in t]))
            linea = f"{video_num} {valores}\n"
            archivo.write(linea)


def calcular_S4(anomalias_predichas, anomalias_reales):

    # Obtener las anomalias predichas y reales
    anomalias_predichas = procesar_anomalias(correspondencias, anomalias)
    anomalias_reales = anomalias_reales()

    # Calcular F1
    TP = len(anomalias_reales)  # True Positives
    FP = len(anomalias_predichas) - TP  # False Positives
    FN = len(anomalias_reales) - TP  # False Negatives
    if 2 * TP + FN + FP == 0:
        F1 = 0
    else:
        F1 = 2 * TP / (2 * TP + FN + FP)

    # Calcular NRMSE
    NRMSE = 0.0
    for clave, ids_objs in anomalias_reales:
        inicio = anomalias_predichas[i][1]
        final = anomalias_predichas[i][2]
        NRMSE += min(math.sqrt(sum([(ti_p - ti_gt) ** 2 for ti_p, ti_gt in zip(t_p, t_gt)]) / TP), 300) / 300

    # Calcular S4
    S4 = F1 * (1 - NRMSE)
    
    return S4

# Llamar a la función para calcular S4
# S4_result = calcular_S4(correspondencias, anomalias)
#print("El valor de S4 es:", S4_result)


# if __name__ == '__main__':
def one_video():
    # labels_path = '/home/usp/Documents/Ana/Track960/detection4/labels/26'
    # video_path = '/home/usp/Documents/Ana/Track960/detection4/26.mp4'
    # labels_path = '/home/usp/Documents/Ana/Track960/detection9/labels/43'
    # video_path = '/home/usp/Documents/Ana/Track960/detection9/43.mp4'
    # labels_path = '/home/usp/EXPERIMENTOS_TRACK4/Experimentos prueba/Deteccion_yolo/labels'
    # video_path = '/home/usp/EXPERIMENTOS_TRACK4/Experimentos prueba/Deteccion_yolo/73_cut1.mp4'
    # labels_path = '/home/usp/Documents/Ana/Track960/detection11/labels/45'
    # video_path = '/home/usp/Documents/Ana/Track960/detection11/45.mp4'
    labels_path = '/home/usp/Documents/Ana/Track960/detection3/labels/23'
    video_path = '/home/usp/Documents/Ana/Track960/detection3/23.mp4'
    labels_path = '/home/usp/Documents/Ana/Track960/detection8/labels/41'
    video_path = '/home/usp/Documents/Ana/Track960/detection8/41.mp4'
 
    rastreo, anomalias_ids = rastreamento_vehiculo_parado(labels_path, video_path, 0.3)
    ch = procesar_anomalias(rastreo, anomalias_ids)
    etiquetas = etiquetas_rastreamento_video(video_path, rastreo)
    # print(ch)
