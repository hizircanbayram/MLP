class optimizer():
    
    def _backwardPropagation(self, predY, groundY, nn_obj):
        '''
        Classes that will inherit this interface has to take the followings into account:
            They can use nn_obj to get the properties of NN object
            They have to return updated weights and biases of the NN in the normal order
                L1_weights, L2_weights, ... in a python list
        '''
        pass