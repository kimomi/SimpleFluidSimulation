from enum import IntEnum 
import taichi as ti

ti.init(arch=ti.gpu)

#Map dimensions
map_width = 100
map_height = 100
map_scale = 6
 
#Block types
class BlockType(IntEnum):
    AIR = 0
    GROUND = 1
    WATER = 2
 
#Data structures
blocks = ti.field(shape=(map_width+2, map_height+2), dtype=ti.i32)

# set original map
for x in range(0, map_width + 2):
    for y in range(0, map_height + 2):
        if x == 0 or y == 0 or x == map_width + 1 or y == map_height + 1:
            blocks[x, y] = BlockType.GROUND
        else:
            blocks[x, y] = BlockType.AIR

@ti.kernel
def simulate_compression():
    for _ in range(1):
        for y in range(0, map_height + 2):
            for x in range(0, map_width + 2):
                if blocks[x, y] != BlockType.WATER: continue # discard border

                blocks[x, y] = BlockType.AIR
                if blocks[x, y - 1] == BlockType.AIR: # down
                    blocks[x, y - 1] = BlockType.WATER
                elif blocks[x - 1, y - 1] == BlockType.AIR and blocks[x + 1, y - 1] == BlockType.AIR: # random leftdown or rightdown
                    blocks[x + ti.i32(ti.math.round(ti.random())) * 2 - 1, y - 1] = BlockType.WATER
                elif blocks[x + 1, y - 1] == BlockType.AIR: # right down
                    blocks[x + 1, y - 1] = BlockType.WATER
                elif blocks[x - 1, y - 1] == BlockType.AIR: # left down
                    blocks[x - 1, y - 1] = BlockType.WATER
                else:
                    check_x = x + ti.i32(ti.math.round(ti.random())) * 2 - 1
                    if blocks[check_x, y] == BlockType.AIR: # random left or right
                        blocks[check_x, y] = BlockType.WATER
                    else:
                        blocks[x, y] = BlockType.WATER

show_map = ti.field(shape=((map_width+2) * map_scale, (map_width+2) * map_scale, 3), dtype=ti.f16)

@ti.kernel
def render():
    for x, y, i in show_map:
        if blocks[x // map_scale, y // map_scale] == BlockType.GROUND:
            show_map[x, y, 0] = 0.85
            show_map[x, y, 1] = 0.8
            show_map[x, y, 2] = 0.1
        elif blocks[x // map_scale, y // map_scale] == BlockType.WATER:
            show_map[x, y, 0] = 0.1
            show_map[x, y, 1] = 0.1
            show_map[x, y, 2] = 1.0
        else:
            show_map[x, y, 0] = 0.0
            show_map[x, y, 1] = 0.0
            show_map[x, y, 2] = 0.0

gui = ti.GUI("fluid sim", res=(show_map.shape[0] ,show_map.shape[1]))

def change_mape(pos, size, type):
    for i in range(-size, size):
        for j in range(-size, size):
            blocks[pos[0] + i, pos[1] + j] = type

def get_input_pos():
    x, y = gui.get_cursor_pos()
    x = ti.math.floor(x * (map_width + 2))
    y = ti.math.floor(y * (map_height + 2))
    return (x, y)

while gui.running:
    simulate_compression()
    gui.get_event()

    if gui.is_pressed(ti.GUI.LMB): # add water
        change_mape(get_input_pos(), 2, BlockType.WATER)    
    if gui.is_pressed(ti.GUI.RMB): # add block
        change_mape(get_input_pos(), 2, BlockType.GROUND)
    if gui.is_pressed(ti.GUI.MMB): # remove
        change_mape(get_input_pos(), 1, BlockType.AIR)

    render()
    gui.set_image(show_map)
    gui.show()