import pattern

def main():

    wave_image = pattern.Wave_2d(260, 346, 20, 30, 1) 
    #wave_image.generate_moving_wave('vertical', 'square')
    wave_image.generate_moving_wave('vertical', 'sine')
    


if __name__=='__main__':
    main()