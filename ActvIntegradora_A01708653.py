# ===============================================================================================================
# File: ActvIntegradora_A01708653.py
# Author: María Fernanda Moreno Gómez A01708653
# Description: Este archivo contiene el código para la simulación de WaTor, la cual simula
#              inicialmente 120 peces y 40 tiburones, los cuales se van moviendo por una cuadrícula, perdiendo
#              energía en cada movimiento (y muriendo a la pérdida de esta) así como el poder devorar los 
#              tiburones a los peces ganando energía. Cuando peces y tiburones tienen la suficiente energía, 
#              pueden tener hijos, y así seguir con la descendencia evitando la extinción.
# To compile: python3 ActvIntegradora_A01708653.py (Tomando en cuenta el uso de consola de Ubuntu)
# ==============================================================================================================

# Importamos las clases que se requieren para manejar los agentes (Agent) y su entorno (Model).
from mesa import Agent, Model
# Debido a que necesitamos que existe un solo agente por celda, elegimos ''SingleGrid''.
from mesa.space import SingleGrid
# Con ''RandomActivation'', hacemos que todos los agentes se activen ''al mismo tiempo''.
from mesa.time import RandomActivation
# Haremos uso de ''DataCollector'' para obtener información de cada paso de la simulación (Pues hay iteraciones)
from mesa.datacollection import DataCollector

# Manejo de números
import numpy as np

# matplotlib lo usaremos crear una animación de cada uno de los pasos del modelo.
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#----------------------------------------------------------------------------------------------------
#----------------------------------------Clase Fish--------------------------------------------------
# En esta clase se representan los agentes Fish (Pez), los cuales tiene un atributo de energía y se
# pueden moverse, dando steps en la cuadrícula, además del poder reproducirse y el poder ser comidos 
#----------------------------------------------------------------------------------------------------
class Fish(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.energy = self.model.fish_initial_energy

    def move(self):
        if self.pos is None:
            return  # No hay acción si la posición es None

        # Encontrar las celdas adyacentes en el agente, que sean ortogonales (no diagonales)
        possible_moves = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        # Se aseguran que sean posiciones ortogonales
        orthogonal_movements = [(x, y) for x, y in possible_moves if (x == self.pos[0] and y != self.pos[1]) or (x != self.pos[0] and y == self.pos[1])]
        # Se filtran aquellas que están vacías
        orthogonal_empty_movements = [movimiento for movimiento in orthogonal_movements if self.model.grid.is_cell_empty(movimiento)]
        # Si hay celdas ortogonales vacías, el pez puede moverse a ellas y pierde 1 de energía de su energía actual
        if orthogonal_empty_movements:
            new_position = self.random.choice(orthogonal_empty_movements)
            # Actualiza la posición del agente en la cuadrícula
            self.model.grid.move_agent(self, new_position)  
            self.energy -= 1
            
    def step(self):
        if self.pos is None:
            return  # No hay acción si la posición es None

        self.move()
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
        else:
            neighbors = self.model.grid.get_neighbors(self.pos, moore=False, include_center=False)
            # Se checan los vecinos, si tiene menos de cuatro vecinos puede avanzar para posiblemente reproducurse (si es que puede)
            if len(neighbors) < 4:
                # Aquí, self.model.random.random() genera un número aleatorio entre 0 y 1, mientras que (1 / self.model.fish_fertility_threshold)
                # evalúa el umbral de fertilidad, ya que puede o no reproducirse (es un umbral, no es obligatoria la reproducción, sino que es probabilidad)
                # Dando una probabilidad del 25% de reproducción al ser 1/4, por ende, si da 1<0.25, no se podrá reproducir, pero su da que 0<0.25, se podrá reproducir
                if self.model.random.random() < (1 / self.model.fish_fertility_threshold):
                    possible_moves = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
                    empty_moves = [move for move in possible_moves if self.model.grid.is_cell_empty(move)]
                    self.energy -= 1
                    if empty_moves:
                        new_position = self.random.choice(empty_moves)
                        # Guardar la posición actual antes de moverse para tener hijos
                        current_position = self.pos
                        self.model.grid.move_agent(self, new_position)  # Moverse a nueva posición

                        # Generar la nueva posición para la descendencia
                        offspring_position = current_position  # La posición de la descendencia es la posición actual del padre antes de moverse
                        if self.model.grid.is_cell_empty(offspring_position):  
                            # Generar un nuevo agente Fish
                            new_fish = Fish(self.model.next_id(), self.model)
                            # Colocarlo en la posición donde se encontraba el padre
                            self.model.grid.place_agent(new_fish, offspring_position)
                            # Asegurar que se cree el agente en el futuro de la simulación
                            self.model.schedule.add(new_fish)

#----------------------------------------------------------------------------------------------------
#----------------------------------------Clase Shark-------------------------------------------------
# En esta clase se representan los agentes Shark (Tiburón), que tienen energía, la capacidad de mo-
# verse por la cuadrícula, comer a los peces y ganar energía, reproducirse o morir por falta de energía
#----------------------------------------------------------------------------------------------------
class Shark(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.energy = self.model.shark_initial_energy

    def move(self):
        # Verificar la posición actual, si es None, no tiene una posición válida y no avanza
        if self.pos is None:
            return  

        # Encontrar las celdas adyacentes en el agente, que sean ortogonales (no diagonales)
        possible_moves = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
        # Se aseguran que sean posiciones ortogonales
        orthogonal_movements = [(x, y) for x, y in possible_moves if (x == self.pos[0] and y != self.pos[1]) or (x != self.pos[0] and y == self.pos[1])]
        # Se filtran aquellas que están vacías
        orthogonal_empty_movements = [movimiento for movimiento in orthogonal_movements if self.model.grid.is_cell_empty(movimiento)]

        # Busca los vecinos en las celdas ortogonales que se encuentre (para podérselos comer)
        fish_neighbors = [agent for agent in self.model.grid.get_neighbors(self.pos, moore=False, include_center=False) if isinstance(agent, Fish)]
        # Si hay peces como vecinos
        if fish_neighbors:
            # Obtiene el primer valor de la lista de vecinos de peces
            fish_agent = fish_neighbors[0]
            # Obtiene la posición de dicho pez
            new_position = fish_agent.pos
            # Se quita al pez de la cuadrícula porque se lo comió  
            self.model.grid.remove_agent(fish_agent)
            # Se actualiza la posición del Shark a donde estaba el Fish que se comió en la Grid (Cuadrícula)
            self.model.grid.move_agent(self, new_position)
            # El tiburón gana energía y se le suma a la que ya tenía  
            self.energy += self.model.shark_gain_energy
        # Si hay celdas disponibles ortogonales sin peces
        elif orthogonal_empty_movements:
            # Se elige una de estas celdas aleatoriamente
            new_position = self.random.choice(orthogonal_empty_movements)
            # Se actualiza la nueva posición del Shark a la celda aleatoria para que se mueva
            self.model.grid.move_agent(self, new_position)  
            # Pierde 1 de energía
            self.energy -= 1
        #Si no hay ni celdas vacías, se queda en su mismo lugar (No hay movimiento)
        else:
            return 

    def step(self):
        self.move()
        # Si la energía del Shark es menor o igual a 0, se remueve de la cuadrícula (Pues se murió)
        if self.energy <= 0:
            self.model.grid.remove_agent(self)
        else:
            # De otra manera, si la energía del Shark es mayor o igual a su umbral de fertilidad (Para poder reproducirse)
            if self.energy >= self.model.shark_fertility_threshold:
                # Se reproduce y se le quita 1 de energía por esta acción
                self.energy -= 1
                # Encontrar las celdas adyacentes en el agente, que sean ortogonales (no diagonales)
                possible_moves = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False)
                # Se buscan las celdas adyacentes ortogonales vacías
                empty_moves = [move for move in possible_moves if self.model.grid.is_cell_empty(move)]
                # Si estas existen
                if empty_moves:
                    #Se escoge aleatoriamente una celda vacía para moverse
                    new_position = self.random.choice(empty_moves)
                    # Guarda la posición actual antes de moverse para tener hijos
                    current_position = self.pos
                    # Se mueve a la nueva posición
                    self.model.grid.move_agent(self, new_position)  

                    # La posición de la descendencia es la posición actual del padre antes de moverse
                    offspring_position = current_position  
                    # Verificar si la celda está vacía
                    if self.model.grid.is_cell_empty(offspring_position):  
                        # Crear un nuevo agente Shark
                        new_shark = Shark(self.model.next_id(), self.model)
                        # Poner al nuevo tiburón en la posición donde estaba el padre
                        self.model.grid.place_agent(new_shark, offspring_position)  
                        # Asegurar que se cree el agente en el futuro de la simulación
                        self.model.schedule.add(new_shark)

#----------------------------------------------------------------------------------------------------
#--------------------------------------Clase WatorGrid-----------------------------------------------
# En esta clase se necesita SingleGrid, de modo que sólo puede haber un agente por celda y busca 
# aquellas celdas vacías en la cuadrícula.
#----------------------------------------------------------------------------------------------------
class WatorGrid(SingleGrid):
    def __init__(self, width, height, torus):
        super().__init__(width, height, torus)
        self.random = np.random.RandomState()

    def find_empty(self):
        while True:
            x = self.random.randint(0, self.width)
            y = self.random.randint(0, self.height)
            if self.is_cell_empty((x, y)):
                return (x, y)

#----------------------------------------------------------------------------------------------------
#--------------------------------------Clase WatorModel-----------------------------------------------
# En esta clase se define el modelo general de la simulación de WaTor, aquí se almacena la configu-
# ración inicial. Crea la cuadrícula, los Fish y Shark (Agentes) de acuerdo a sus atributos. Avanza
# la simulación en el tiempo con step y guarda los datos de cada paso en DataCollector
#----------------------------------------------------------------------------------------------------
class WatorModel(Model):
    def __init__(self, width, height, fish_count, shark_count, fish_initial_energy, shark_initial_energy, fish_fertility_threshold, shark_fertility_threshold, shark_gain_energy):
        self.num_agents = fish_count + shark_count
        self.grid = WatorGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters={"Grid": get_grid})
        self.fish_initial_energy = fish_initial_energy
        self.shark_initial_energy = shark_initial_energy
        self.fish_fertility_threshold = fish_fertility_threshold
        self.shark_fertility_threshold = shark_fertility_threshold
        self.shark_gain_energy = shark_gain_energy
        self.current_id = 0

        # Crear agentes Fish
        for i in range(fish_count):
            x, y = self.grid.find_empty()
            fish = Fish(self.current_id, self)
            self.grid.place_agent(fish, (x, y))
            self.schedule.add(fish)
            self.current_id += 1

        # Crear agentes Shark
        for i in range(shark_count):
            shark = Shark(self.current_id, self)
            x, y = self.grid.find_empty()
            self.grid.place_agent(shark, (x, y))
            self.schedule.add(shark)
            self.current_id += 1

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


#-----------------------------Función get_grid----------------------------------------
# Sirve para representar de manera matricial la matrícula, donde por cada posición en
# la cuadrícula, le corresponde un número, siendo 1 para Fish y 2 para Shark, de lo
# contrario, no tienen valor y son el mar.  
#-------------------------------------------------------------------------------------
def get_grid(model):
    grid = np.zeros((model.grid.width, model.grid.height))
    for agent in model.schedule.agents:
        if agent.pos is not None:
            if isinstance(agent, Fish):
                grid[agent.pos[0]][agent.pos[1]] = 1
            elif isinstance(agent, Shark):
                grid[agent.pos[0]][agent.pos[1]] = 2
    return grid

# Parámetros del modelo
GRID_SIZE = 75
GRID_HEIGHT = 50
FISH_COUNT = 120
SHARK_COUNT = 40
FISH_INITIAL_ENERGY = 20
SHARK_INITIAL_ENERGY = 3
FISH_FERTILITY_THRESHOLD = 4
SHARK_FERTILITY_THRESHOLD = 12
SHARK_GAIN_ENERGY = 1
MAX_ITERATION = 300

# Creación del modelo
model = WatorModel(
    width=GRID_SIZE,
    height=GRID_HEIGHT,
    fish_count=FISH_COUNT,
    shark_count=SHARK_COUNT,
    fish_initial_energy=FISH_INITIAL_ENERGY,
    shark_initial_energy=SHARK_INITIAL_ENERGY,
    fish_fertility_threshold=FISH_FERTILITY_THRESHOLD,
    shark_fertility_threshold=SHARK_FERTILITY_THRESHOLD,
    shark_gain_energy=SHARK_GAIN_ENERGY
)

# Ejecutar el modelo
for i in range(MAX_ITERATION):
    model.step()

# Graficamos la información usando `matplotlib`
all_grid = model.datacollector.get_model_vars_dataframe()

# Definir colores: Naranja para Pez, Gris para Tiburón y Azul para las celdas vacía que representan el mar
blue_color = (0.53, 0.81, 0.92)  # Color azul claro
orange_color = (1.0, 0.75, 0.42)  # Color naranja pastel
gray_color = (0.5, 0.5, 0.5)  # Color gris

# Visualización de la simulación
fig, axis = plt.subplots(figsize=(6, 6))
axis.set_xticks([])
axis.set_yticks([])

patch = plt.imshow(all_grid.iloc[0][0], cmap = plt.cm.colors.ListedColormap([blue_color, orange_color, gray_color]))
def animate(i):
    patch.set_data(all_grid.iloc[i][0])
anim = animation.FuncAnimation(fig, animate, frames= MAX_ITERATION)

anim.save(filename="ActvIntegradora_A01708653.mp4")

