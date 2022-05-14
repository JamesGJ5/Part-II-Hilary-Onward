# Okay, for reading from /media/rob/hdd1/james-gj/forReport/2022-04-29/cosmo.txt to be efficient, I am going to have 
# to catalogue things quite well. Things to be catalogued for a given aberration set:
# 
# - Time at which aberration constants were obtained
# 
# - Aberration in question
# 
# - Magnitude of each aberration constant/unit
# - Unit of aformenetioned magnitude
# - Value in brackets next to magnitude of each aberration constant, presumably the error
# 
# - Angle of each aberration constant/degree
# - Unit of aforementioned angle (degree)
# - Value in brackets next to angle of each aberration constant, presumably the error
# 
# - pi/4 limit in nm
# 
# Going to save these quantities to a dictionary for each aberration calculation result, all of these dictionaries being 
# values in a larger dictionary, and save the larger dictionary using the method in 
# https://www.adamsmith.haus/python/answers/how-to-save-a-dictionary-to-a-file-in-python

import pickle

write = False

if write:

    with open('/media/rob/hdd1/james-gj/forReport/2022-04-29/cosmo.txt', 'r') as f:

        # Initialising dictionary where key is index of aberration calculation result and value is aberration calculation 
        # result (containing results for all aberrations)
        dAllResultSets = {}
        aberList = [
            'Aberration', 'O2', 'A2', 'P3', 'A3', 'O4', 'Q4', 'A4', 'P5', 'R5', 'A5', 'O6', 'Q6', 'S6', 'A6', 'P7', 'R7', 
            'T7', 'A7'
        ]

        resultSetIndex = 0

        # Initialising dictionary where key is name of parameter of aberration result (e.g. aberration symbol, 
        # aberration magnitude/unit, time)
        dSingleResultSet = {}

        for line in f:

            if any(aber in line for aber in aberList):

                # Convert parentheses in the line to whitespace so that the split method can be called with only the 
                # whitespace character as an argument
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                
                # Return the segments of string delimited by whitespace as a list
                lineSegmentList = line.split()

                if 'Aberration' in lineSegmentList:

                    # print(True)
                    dSingleResultSet['timeString'] = lineSegmentList[9]

                else:

                    dSingleResultSet[lineSegmentList[0]] = {
                        
                        'mag': lineSegmentList[1],
                        'magUnit': lineSegmentList[2],
                        'magError': lineSegmentList[3],
                        'angle': lineSegmentList[4],
                        'angleUnit': 'degree',
                        'angleError': lineSegmentList[5],
                        'pi/4Limit': lineSegmentList[6],
                        'pi/4LimitUnit': 'nm'
                    }

                    if 'A7' in line:

                        dAllResultSets[resultSetIndex] = dSingleResultSet
                        resultSetIndex += 1

                        dSingleResultSet = {}

        # print(dAllResultSets)

        with open('/home/james/VSCode/currentPipelines/aberCalcResults_2022_04_29_OPQ.pkl', 'wb') as f2:

            pickle.dump(dAllResultSets, f2)

with open('/home/james/VSCode/currentPipelines/aberCalcResults_2022_04_29_OPQ.pkl', 'rb') as f3:

    output = pickle.load(f3)
    print(output)
    print('\n\n\n')
    print(output[5])