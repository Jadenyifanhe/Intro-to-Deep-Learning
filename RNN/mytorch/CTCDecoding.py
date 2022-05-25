import numpy as np


def clean_path(path):

    """ utility function that performs basic text cleaning on path """

    # No need to modify
    path = str(path).replace("'","")
    path = path.replace(",","")
    path = path.replace(" ","")
    path = path.replace("[","")
    path = path.replace("]","")

    return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path

        for seq in range(len(y_probs[0])):
            for p in range(len(y_probs[0][seq])):
                num = 0
                highest = y_probs[0][seq][p]
                for i in range(len(y_probs)):
                    if y_probs[i][seq][p] > highest:
                        highest = y_probs[i][seq][p]
                        num = i
                path_prob = path_prob * y_probs[num][seq][p]
                if num == blank:
                    continue
                blank = num
                if num == 0:
                    continue
                else:
                    decoded_path.append(self.symbol_set[num - 1])

        decoded_path = clean_path(decoded_path)

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        decoded_path = []
        sequences = [[list(), 1.0]]
        ordered = None

        forward_path, merged_path_scores = None, None

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return

        def InitializePaths(symbol_set, y):
            initialBlankPathScore, initialPathScore = dict(), dict()
            path = ""
            initialBlankPathScore[path] = y[blank]
            initialPathsWithFinalBlank = set(path)
            initialPathsWithFinalBlank.add(path)

            initialPathsWithFinalSymbol = set()
            for i in range(len(symbol_set)):
                path = symbol_set[i]
                initialPathScore[path] = y[i + 1]
                initialPathsWithFinalSymbol.add(path)

            return initialPathsWithFinalBlank, initialPathsWithFinalSymbol, initialBlankPathScore, initialPathScore

        def ExtendWithBlank(pathsWithTerminalBlank, pathsWithTerminalSymbol, y):
            updatedPathsWithTerminalBlank = set()
            updatedBlankPathScore = dict()

            for path in pathsWithTerminalBlank:
                updatedPathsWithTerminalBlank.add(path)
                updatedBlankPathScore[path] = blankPathScore[path] * y[blank]

            for path in pathsWithTerminalSymbol:
                if path in updatedPathsWithTerminalBlank:
                    updatedBlankPathScore[path] += pathScore[path] * y[blank]
                else:
                    updatedPathsWithTerminalBlank.add(path)
                    updatedBlankPathScore[path] = pathScore[path] * y[blank]

            return updatedPathsWithTerminalBlank, updatedBlankPathScore

        def ExtendWithSymbol(pathsWithTerminalBlank, pathsWithTerminalSymbol, symbolSet, y):
            updatedPathsWithTerminalSymbol = set()
            updatedPathScore = dict()

            for path in pathsWithTerminalBlank:
                for i in range(len((symbolSet))):
                    newpath = path + symbolSet[i]
                    updatedPathsWithTerminalSymbol.add(newpath)
                    updatedPathScore[newpath] = blankPathScore[path] * y[i + 1]

            for path in pathsWithTerminalSymbol:
                for i, c in enumerate(symbolSet):
                    if c == path[-1]:
                        newpath = path
                    else:
                        newpath = path + c
                    if newpath in updatedPathsWithTerminalSymbol:
                        updatedPathScore[newpath] += pathScore[path] * y[i + 1]
                    else:
                        updatedPathsWithTerminalSymbol.add(newpath)
                        updatedPathScore[newpath] = pathScore[path] * y[i + 1]

            return updatedPathsWithTerminalSymbol, updatedPathScore

        def Prune(pathsWithTerminalBlank, pathsWithTerminalSymbol, blankPathScore, pathScore, beamWidth):
            prunedBlankPathScore, prunedPathScore = dict(), dict()
            scorelist = []
            for p in pathsWithTerminalBlank:
                scorelist.append(blankPathScore[p])

            for p in pathsWithTerminalSymbol:
                scorelist.append(pathScore[p])

            scorelist.sort(reverse=True)
            if beamWidth < len(scorelist):
                cutoff = scorelist[beamWidth]
            else:
                cutoff = scorelist[-1]

            prunedPathsWithTerminalBlank = set()
            for p in pathsWithTerminalBlank:
                if blankPathScore[p] > cutoff:
                    prunedPathsWithTerminalBlank.add(p)
                    prunedBlankPathScore[p] = blankPathScore[p]

            prunedPathsWithTerminalSymbol = set()
            for p in pathsWithTerminalSymbol:
                if pathScore[p] > cutoff:
                    prunedPathsWithTerminalSymbol.add(p)
                    prunedPathScore[p] = pathScore[p]

            return prunedPathsWithTerminalBlank, prunedPathsWithTerminalSymbol, prunedBlankPathScore, prunedPathScore

        def MergeIdenticalPaths(pathsWithTerminalBlank, blankPathScore, pathsWithTerminalSymbol, pathScore):
            mergedPaths = pathsWithTerminalSymbol
            finalPathScore = pathScore

            for p in pathsWithTerminalBlank:
                if p in mergedPaths:
                    finalPathScore[p] += blankPathScore[p]
                else:
                    mergedPaths.add(p)
                    finalPathScore[p] = blankPathScore[p]

            return mergedPaths, finalPathScore

        blank = 0
        pathScore, blankPathScore = dict(), dict()
        y = y_probs[:, :, 0]

        newPathsWithTerminalBlank, newPathsWithTerminalSymbol, newBlankPathScore, newPathScore = InitializePaths(self.symbol_set, y[:, 0])

        for t in range(1, y.shape[1]):
            pathsWithTerminalBlank, pathsWithTerminalSymbol, blankPathScore, pathScore = Prune(newPathsWithTerminalBlank,
                                                                                               newPathsWithTerminalSymbol,
                                                                                               newBlankPathScore,
                                                                                               newPathScore,
                                                                                               self.beam_width)
            newPathsWithTerminalBlank, newBlankPathScore = ExtendWithBlank(pathsWithTerminalBlank,
                                                                           pathsWithTerminalSymbol,
                                                                           y[:, t])
            newPathsWithTerminalSymbol, newPathScore = ExtendWithSymbol(pathsWithTerminalBlank,
                                                                        pathsWithTerminalSymbol,
                                                                        self.symbol_set, y[:, t])

        mergedPaths, finalPathScore = MergeIdenticalPaths(newPathsWithTerminalBlank,
                                                          newBlankPathScore,
                                                          newPathsWithTerminalSymbol,
                                                          newPathScore)

        forward_path = max(finalPathScore, key=finalPathScore.get)
        merged_path_scores = finalPathScore

        return forward_path, merged_path_scores
