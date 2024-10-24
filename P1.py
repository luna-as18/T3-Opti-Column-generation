from gurobipy import *
import time
import json



# Conjuntos
dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
franjas = ['00am-2am', '2am-4am', '4am-6am', '6am-8am', '8am-10am', '10am-12pm', 
           '12pm-2pm', '2pm-4pm', '4pm-6pm', '6pm-8pm', '8pm-10pm', '10pm-12am']
D = [i for i in range(len(dias))]
F = [i for i in range(len(franjas))]


# Parámetro aij (requerimientos mínimos de equipos por franja horaria y día)
aij = [
    [11, 11, 24, 55, 68, 80, 88, 75, 79, 93, 67, 15],
    [12, 12, 25, 50, 70, 85, 90, 80, 75, 93, 65, 15],
    [11, 11, 24, 55, 68, 80, 88, 75, 79, 93, 67, 15],
    [15, 15, 27, 60, 70, 85, 93, 78, 78, 90, 72, 15],
    [15, 20, 35, 64, 77, 86, 92, 86, 75, 91, 76, 15],
    [15, 20, 35, 64, 77, 86, 92, 86, 75, 91, 76, 15],
    [10, 10, 30, 60, 68, 82, 85, 77, 79, 95, 74, 10]
]

# Parámetro cij (costos por equipo asignado por franja horaria y día)
cij = [
    [100, 100, 100, 66, 66, 66, 66, 72, 72, 100, 100, 100],
    [100, 100, 100, 66, 66, 66, 66, 72, 72, 100, 100, 100],
    [100, 100, 100, 66, 66, 66, 66, 72, 72, 100, 100, 100],
    [100, 100, 100, 66, 66, 66, 66, 72, 72, 100, 100, 100],
    [100, 100, 100, 66, 66, 66, 66, 72, 72, 100, 100, 100],
    [100, 100, 100, 66, 66, 66, 66, 72, 72, 100, 100, 100],
    [130, 130, 130, 84, 84, 84, 84, 97, 97, 130, 130, 130]
]

def costsSchedule(Phij, cij):
    costos =[]
    for k in range(len(Phij)):
        costos_k = 0
        for i in range(len(Phij[k])):
            for j in range(len(cij[i])):
                costos_k += Phij[k][i][j] * cij[i][j]
        costos.append(costos_k)
    return costos

def costFranja(xij, cij):
    costo = 0
    for i in range(len(xij)):
        for j in range(len(cij[i])):
            costo += xij[i][j] * cij[i][j]
    return costo


import numpy as np

def generate_single_filled_schedules():
    schedules = {}
    schedule_count = 0
    
    # Generar horarios en los que solo se asigna un equipo en una franja y día únicos
    for day in range(7):  # 7 días
        for franja in range(12):  # 12 franjas horarias
            # Crear un horario vacío para todos los días
            schedule = [[0] * 12 for _ in range(7)]
            
            # Asignar un equipo solo a la combinación de día y franja actuales
            schedule[day][franja] = 1
            
            # Guardar el horario en el diccionario con un identificador único
            schedules[schedule_count] = schedule
            schedule_count += 1
    
    return schedules

# Generar los horarios
Columnas = generate_single_filled_schedules()

# Medir tiempo de la relajación
start_time = time.time()
start_cpu_time = time.process_time()

Ch = costsSchedule(Columnas, cij)
num_horarios = len(Columnas)
 
# Crear el modelo

ModelMP = Model("Master Problem")
ModelMP.setParam('OutputFlag', 0)

# Crear la variable h
h = []
for j in Columnas.keys():
    #obj_j = costFranja(Columnas[j], cij)
    h.append(ModelMP.addVar(vtype=GRB.CONTINUOUS, obj = 100000000,  name=f"h_{j}"))

# Añadir restricciones para asegurar que se cubran los requerimientos mínimos de equipos por día y franja
demCtr = []
for i in D:
    for j in F:
        demCtr.append(ModelMP.addConstr(quicksum(h[k] * Columnas[k][i][j] for k in range(num_horarios)) >= aij[i][j], 
                        name=f"req_min_equipos_{i}_{j}"))

# Crear la función objetivo
ModelMP.ModelSense = GRB.MINIMIZE

#Parametros
ModelAUX = Model("Auxiliar Problem")
ModelAUX.Params.OutputFlag = 0

# Crear la variable x que indica si se asigna la franja f del dia d al horario
x = ModelAUX.addVars(len(D), len(F), vtype=GRB.BINARY, name="x")

# Crear la variable y que tiene la cantidad de variables de los días y que sea binaria
y = ModelAUX.addVars(len(D), vtype=GRB.BINARY, name="y")

# Restricción para asegurar que el equipo trabaje exactamente 4 franjas por día
for i in D:
    ModelAUX.addConstr(quicksum(x[i,j] for j in F) == 4 * y[i], name=f"exactly_4_franjas_{i}")

# Restricción Asegurar que el equipo trabaje al menos 32 horas entre semana:
ModelAUX.addConstr(quicksum(x[i,j] for i in D for j in F) >= 16, name="min_32_horas_entre_semana")

# Restricción Asegurar que el equipo trabaje un maximo de 32  horas entre semana:
ModelAUX.addConstr(quicksum(x[i,j] for i in D for j in F) <= 20, name="max_40_horas_entre_semana")

# Restricción para asegurar que el equipo trabaje un máximo de un día por fin de semana
ModelAUX.addConstr(y[5] + y[6] <= 1, name="max_1_dia_fin_semana")

# Restricción Asegurar que el equipo no trabaje más de dos días consecutivos
for i in range(len(D)-2):
    ModelAUX.addConstr(y[i]+y[i+1]+y[i+2] <= 2, name=f"max_2_dias_consecutivos_{i}")

iteracion = 0  
# Extraer los horarios finales y las cantidades de equipos de la solución relajada
resultados_relajados = {}
columnas_finales = {}

while True:
    
    ModelMP.optimize()
    # Guardar el modelo MP y la solución
    ModelMP.write(f"MasterProblem_{iteracion}.lp")
    ModelMP.write(f"MasterProblem_{iteracion}.sol")

    print(f"FO Master: {ModelMP.objVal} --> #Cols: {num_horarios}")
    # Obtener las duales
    Duals = ModelMP.getAttr("Pi", ModelMP.getConstrs())
    duals = {}

    indexDuals= 0
    for i in D:
        for j in F:
            duals[(i, j)] = Duals[indexDuals]
            indexDuals += 1

    ModelAUX.setObjective(quicksum(x[i,j]*(cij[i][j] - duals[i,j]) for i in D for j in F), GRB.MINIMIZE)

    ModelAUX.optimize()
    ModelAUX.setParam('OutputFlag', 0)
    # Guardar el modelo AUX y la solución
    ModelAUX.write(f"AuxProblem_{iteracion}.lp")
    ModelAUX.write(f"AuxProblem_{iteracion}.sol")
    
    costo_reducido = ModelAUX.getObjective().getValue()

    # Break or continue
    if costo_reducido >= 0:
        print(costo_reducido)
        print(f"FO Aux (Costo Reducido): {ModelAUX.objVal} ")
        print("\nColumn generation stops !\n")
        break
    else:
        num_horarios += 1
        print(ModelAUX.getAttr("X"))
        col_vals = ModelAUX.getAttr("x", x)
         # Crear listas para cada día y reemplazar -0.0 por 0.0
        # Crear una lista de días
        dias = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']

        # Inicializar una lista para almacenar los horarios de cada día
        nuevo_horario = []
        nueva_columna=[]


# Usar un for loop para crear las listas de franjas horarias para cada día
        for i in range(len(dias)):
          horario_dia = [col_vals[i, j] for j in F]  # Extraer las franjas horarias del día i
          nuevo_horario.append(horario_dia)  # Agregar el horario del día i a nuevo_horario
          nueva_columna += horario_dia

        Columnas[num_horarios - 1]= nuevo_horario
        
        obj_j = costFranja(nuevo_horario, cij)
        newCol = Column(nueva_columna, demCtr)
        ModelMP.addVar(vtype = GRB.CONTINUOUS, obj = obj_j, column = newCol,  name= f"h_{num_horarios - 1}")
        ModelMP.update()
        # Agregar el nuevo costo a la lista de costos
        print(f"Costo del turno: {costFranja(nuevo_horario, cij)}")
        Ch.append(costFranja(nuevo_horario, cij))
        print(f"Iteración {num_horarios - 6}")
        print(f"FO Aux (Costo Reducido): {ModelAUX.objVal} ")
    
    iteracion +=1

if ModelMP.status == GRB.OPTIMAL:
    for v in ModelMP.getVars():
        print(f'{v.varName}: {v.x}')
else:
    print("No se encontró una solución óptima.")
    
num_horarios = len(Columnas)

# Calcular tiempo transcurrido
relaxed_time = time.time() - start_time
print(f"Tiempo transcurrido para la solución relajada: {relaxed_time:.2f} segundos")
# Código cuyo tiempo de CPU quieres medir
end_cpu_time = time.process_time()

cpu_time = end_cpu_time - start_cpu_time
print(f"El tiempo de CPU utilizado fue: {cpu_time} segundos")
print(f"El número total de columnas generadas es: {iteracion + 1}")

#--------- GRAFICAR LA SOLUCIÓN ------------------------------------------------------

import json

# Extraer los horarios finales y las cantidades de equipos de la solución relajada
resultados_relajados = {}
columnas_finales = {}

# Asumiendo que ya has generado las columnas en `Columnas`
for col_idx in range(87, len(Columnas)):  # Desde la columna 87 en adelante
    # Aplanar la columna (matriz de horarios)
    columna_aplanada = [item for sublist in Columnas[col_idx] for item in sublist]
    
    # Añadir al diccionario el nombre de la columna y su contenido aplanado
    columnas_finales[f"h_{col_idx}"] = columna_aplanada

# Guardar el resultado en un archivo JSON para las columnas aplanadas
with open("columnas_finales_87.json", "w") as outfile:
    json.dump(columnas_finales, outfile)

print(f"Archivo JSON 'columnas_finales_87.json' generado con éxito.")

# Crear otro archivo JSON con el número de equipos por columna
equipos_por_columna = {}
for v in ModelMP.getVars():
    if v.varName.startswith('h_') and v.x > 0:
        equipos_por_columna[v.varName] = v.x  # Guardar el nombre de la columna y el número de equipos

# Guardar el resultado en un archivo JSON para el número de equipos por columna
with open("equipos_por_columna.json", "w") as outfile:
    json.dump(equipos_por_columna, outfile)

print(f"Archivo JSON 'equipos_por_columna.json' generado con éxito.")


#--------- RESOLUCIÓN ENTERA ------------------------------
for v in ModelMP.getVars():
    v.setAttr("vtype", GRB.INTEGER)

# Establecer los parámetros para MIPGap y Timelimit
ModelMP.setParam(GRB.Param.MIPGap, 0.01)      # Establecer MIPGap en 1.0%
ModelMP.setParam(GRB.Param.TimeLimit, 3600)   # Establecer Timelimit en 3600 segundos (1 hora)

# Actualizar el modelo
ModelMP.update()

# Optimizar el modelo
ModelMP.optimize()

# Imprimir resultados
if ModelMP.status == GRB.OPTIMAL or ModelMP.status == GRB.TIME_LIMIT:
    print(f"Cantidad de Auxiliares~ Entero y Costo Total {ModelMP.objVal}")
    for v in ModelMP.getVars():
        if v.x > 0:
            print(f"TURNO: {v.varName} --> No. Equipos: {v.x}")
else:
    print("No se encontró una solución óptima dentro del límite de tiempo o con el gap permitido.")

print()
