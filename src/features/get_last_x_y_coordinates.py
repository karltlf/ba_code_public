def get_last_x_y_coordinates(x_y_tuples):
    # write a doc comment
    '''
    Returns a list of the last entries of a x-y tuple list. x and y values are matched in order to create two lists of x and y coordinates.

    :param x_y_tuples: A list of x-y tuples, as delivered by `src.features.get_x_y_tuples`.
    :return last_x_y_coordinates: A tuple including two lists of the last entries of the 1) x and 2) y coordinates
    
        '''
   # create a list of all the first x-y coordinates of each row and a list for the last entries
    last_x_y = [x_y[-1] for x_y in x_y_tuples]

    # make a list of all x coordinates and a list of all y coordinates using both lists
    x_values_last = [x for x,y in last_x_y]
    y_values_last = [y for x,y in last_x_y]


    return (x_values_last, y_values_last)