import time
import itertools
import numpy as np
import copy

class NeuroEvolution:

    class EvolvableNetwork:

        def relu(self, x):
            return np.maximum(x, 0)

        def tanh(self, x):
            return np.tanh(x)

        def logistic(self, x):
            return 1 / (1 + np.exp(-x))

        def unit(self, x):
            return np.heaviside(x, 0)

        def softmax(self, x):
            return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

        def gaussian(self, x):
            return np.exp(- (x ** 2))

        def purlin(self, x):
            return x

        def __init__(self, layer_nodes, num_input, num_output, activation_function='relu', output_activation='purlin',
                     initialize=True, use_links=False, use_recurrent=False, recurrent_activation='relu'):

            self.activation_function = []
            self.use_recurrent = use_recurrent
            if activation_function == 'relu':
                self.activation_function_name = 'relu'
                for i in range(0, len(layer_nodes)+ 1):
                    self.activation_function.append(self.relu)
            elif activation_function == 'tanh':
                self.activation_function_name = 'tanh'
                for i in range(0, len(layer_nodes)+ 1):
                    self.activation_function.append(self.tanh)
            elif activation_function == 'logistic':
                self.activation_function_name = 'logistic'
                for i in range(0, len(layer_nodes)+ 1):
                    self.activation_function.append(self.logistic)
            elif activation_function == 'unit':
                self.activation_function_name = 'unit'
                for i in range(0, len(layer_nodes)+ 1):
                    self.activation_function.append(self.unit)
            elif activation_function == 'purlin':
                self.activation_function_name = 'purlin'
                for i in range(0, len(layer_nodes)+ 1):
                    self.activation_function.append(self.purlin)
            elif activation_function == 'softmax':
                self.activation_function_name = 'softmax'
                for i in range(0, len(layer_nodes)+ 1):
                    self.activation_function.append(self.softmax)
            elif activation_function == 'gaussian':
                self.activation_function_name = 'gaussian'
                for i in range(0, len(layer_nodes)+ 1):
                    self.activation_function.append(self.gaussian)
            else:
                self.activation_function_name = []
                for fun in activation_function:
                    if fun == 'relu':
                        self.activation_function_name.append("relu")
                        self.activation_function.append(self.relu)
                    elif fun == 'tanh':
                        self.activation_function_name.append("tanh")
                        self.activation_function.append(self.tanh)
                    elif fun == 'logistic':
                        self.activation_function_name.append("logistic")
                        self.activation_function.append(self.logistic)
                    elif fun == 'unit':
                        self.activation_function_name.append("unit")
                        self.activation_function.append(self.unit)
                    elif fun == 'purlin':
                        self.activation_function_name.append("purlin")
                        self.activation_function.append(self.purlin)
                    elif fun == 'softmax':
                        self.activation_function_name.append("softmax")
                        self.activation_function.append(self.softmax)
                    elif fun == 'gaussian':
                        self.activation_function_name.append("gaussian")
                        self.activation_function.append(self.gaussian)

            self.output_activation_name = output_activation
            if output_activation == 'relu':
                self.output_activation = self.relu
            elif output_activation == 'tanh':
                self.output_activation = self.tanh
            elif output_activation == 'logistic':
                self.output_activation = self.logistic
            elif output_activation == 'unit':
                self.output_activation = self.unit
            elif output_activation == 'purlin':
                self.output_activation = self.purlin
            elif output_activation == 'softmax':
                self.output_activation = self.softmax
            elif output_activation == 'gaussian':
                self.output_activation = self.gaussian

            self.recurrent_activation_name = recurrent_activation
            if recurrent_activation == 'relu':
                self.recurrent_activation = self.relu
            elif recurrent_activation == 'tanh':
                self.recurrent_activation = self.tanh
            elif recurrent_activation == 'logistic':
                self.recurrent_activation = self.logistic
            elif recurrent_activation == 'unit':
                self.recurrent_activation = self.unit
            elif recurrent_activation == 'purlin':
                self.recurrent_activation = self.purlin
            elif recurrent_activation == 'softmax':
                self.recurrent_activation = self.softmax
            elif recurrent_activation == 'gaussian':
                self.recurrent_activation = self.gaussian

            self.layer_count = len(layer_nodes)
            self.layer_nodes = layer_nodes
            self.num_input = num_input
            self.num_output = num_output
            self.layers_weights = []
            self.biases_weights = []
            self.use_links = use_links
            if use_links:
                self.layers_links = []
                self.biases_links = []

            if not initialize:
                return

            limit = np.sqrt(6/(num_output+num_output))

            if use_recurrent:
                self.recurrent_weight = np.random.uniform(-limit, limit, layer_nodes[len(layer_nodes)-1] * layer_nodes[len(layer_nodes)-1]).reshape(layer_nodes[len(layer_nodes)-1],
                                                                                          layer_nodes[len(layer_nodes)-1])
                self.recurrent_bias = np.random.uniform(-limit, limit, layer_nodes[len(layer_nodes)-1])
                if use_links:
                    self.recurrent_w_link = np.random.uniform(-limit, limit, layer_nodes[len(layer_nodes)-1]*layer_nodes[len(layer_nodes)-1]).reshape(layer_nodes[len(layer_nodes)-1],layer_nodes[len(layer_nodes)-1])
                    self.recurrent_b_link = np.random.uniform(-limit, limit, layer_nodes[len(layer_nodes) - 1])


            self.layers_weights.append(
                np.random.uniform(-limit, limit, num_input * layer_nodes[0]).reshape(num_input, layer_nodes[0]))

            self.biases_weights.append(np.random.uniform(-limit, limit, layer_nodes[0]))

            if use_links:
                self.layers_links.append(
                    np.random.uniform(-limit, limit, num_input * layer_nodes[0]).reshape(num_input, layer_nodes[0]))

                self.biases_links.append(np.random.uniform(-limit, limit, layer_nodes[0]))
            for i in range(1, self.layer_count):
                self.layers_weights.append(
                    np.random.uniform(-limit, limit, layer_nodes[i - 1] * layer_nodes[i]).reshape(layer_nodes[i - 1],
                                                                                          layer_nodes[i]))
                self.biases_weights.append(np.random.uniform(-limit, limit, layer_nodes[i]).reshape(1, layer_nodes[i]))
                if use_links:
                    self.layers_links.append(
                        np.random.uniform(-limit, limit, layer_nodes[i - 1] * layer_nodes[i]).reshape(layer_nodes[i - 1],
                                                                                                layer_nodes[i]))
                    self.biases_links.append(np.random.uniform(-limit, limit, layer_nodes[i]).reshape(1, layer_nodes[i]))
            self.layers_weights.append(
                np.random.uniform(-limit, limit, layer_nodes[self.layer_count - 1] * num_output).reshape(
                    layer_nodes[self.layer_count - 1], num_output))
            self.biases_weights.append(np.random.uniform(-limit, limit, num_output).reshape(1, num_output))
            if use_links:
                self.layers_links.append(
                    np.random.uniform(-limit, limit, layer_nodes[self.layer_count - 1] * num_output).reshape(
                        layer_nodes[self.layer_count - 1], num_output))
                self.biases_links.append(np.random.uniform(-limit, limit, num_output).reshape(1, num_output))

        def predict(self, x):
            if self.use_recurrent:
                prev = np.zeros(shape=(1,self.layer_nodes[len(self.layer_nodes)-1]))
                total_output = []
                if self.use_links:
                    for val in x:
                        output = self.activation_function[0](
                            np.dot(val, self.layers_weights[0] * self.unit(self.layers_links[0])) +
                            self.biases_weights[0] * self.unit(self.biases_links[0]))
                        for i in range(1, self.layer_count + 1):
                            if i == self.layer_count:
                                output_2 = output + np.dot(prev, self.recurrent_weight*self.unit(self.recurrent_w_link)) + self.recurrent_bias*self.unit(self.recurrent_b_link)
                                output = (np.dot(output_2, self.layers_weights[i] * self.unit(self.layers_links[i])) +
                                          self.biases_weights[i] * self.unit(self.biases_links[i]))
                            else:
                                output = self.activation_function[i](
                                    np.dot(output, self.layers_weights[i] * self.unit(self.layers_links[i])) +
                                    self.biases_weights[i] * self.unit(self.biases_links[i]))
                        prev = self.recurrent_activation(output_2)
                        total_output.append(output[0][0])
                else:
                    for val in x:
                        output = self.activation_function[0](np.dot(val, self.layers_weights[0]) + self.biases_weights[0])
                        for i in range(1, self.layer_count + 1):
                            if i == self.layer_count:
                                output_2 = output + np.dot(prev, self.recurrent_weight) + self.recurrent_bias
                                output = (np.dot(output_2, self.layers_weights[i]) + self.biases_weights[i])
                            else:
                                output = self.activation_function[i](
                                    np.dot(output, self.layers_weights[i]) + self.biases_weights[i])
                        prev = self.recurrent_activation(output_2)
                        total_output.append(output[0][0])
                if self.num_output == 1:
                    return self.output_activation(np.asarray(total_output)).reshape(len(x), )
                return self.output_activation(np.asarray(total_output))

            if self.use_links:
                output = self.activation_function[0](np.dot(x, self.layers_weights[0] * self.unit(self.layers_links[0])) +
                                                  self.biases_weights[0] * self.unit(self.biases_links[0]))
                for i in range(1, self.layer_count + 1):
                    if i == self.layer_count:
                        output = (np.dot(output, self.layers_weights[i] * self.unit(self.layers_links[i])) +
                            self.biases_weights[i] * self.unit(self.biases_links[i]))
                    else:
                        output = self.activation_function[i](
                            np.dot(output, self.layers_weights[i] * self.unit(self.layers_links[i])) +
                            self.biases_weights[i] * self.unit(self.biases_links[i]))
                if self.num_output == 1:
                    return self.output_activation(output).reshape(len(x), )
                return self.output_activation(output)
            else:
                output = self.activation_function[0](np.dot(x, self.layers_weights[0]) + self.biases_weights[0])
                for i in range(1, self.layer_count + 1):
                    if i == self.layer_count:
                        output = (np.dot(output, self.layers_weights[i]) + self.biases_weights[i])
                    else:
                        output = self.activation_function[i](
                            np.dot(output, self.layers_weights[i]) + self.biases_weights[i])
                if self.num_output == 1:
                    return self.output_activation(output).reshape(len(x), )
                return self.output_activation(output)

    def _mse(self, actual, prediction):
        return np.sum(np.square(np.subtract(actual, prediction)).mean())

    def _r_2(self, actual, prediction):
        ss_resid = np.sum(np.square(np.subtract(actual, prediction)))
        ss_total = np.sum(np.square(np.subtract(actual, np.mean(actual))))
        return 1 - (float(ss_resid))/ss_total

    def _mae(self, actual, prediction):
        return np.sum(np.abs(np.subtract(actual, prediction)).mean())

    def __init__(self, layer_nodes, num_input, num_output, default_layer_activation="relu", default_output_activation='purlin'):
        self.layer_nodes = layer_nodes
        self.num_input = num_input
        self.num_output = num_output
        self.default_layer_activation = default_layer_activation
        self.default_output_activation = default_output_activation

    def _get_num_links_used(self, model):
        num = 0
        num2 = -1
        for i in range(0, model.layer_count+1):
            num += np.sum(model.unit(model.layers_links[i]))+np.sum(model.unit(model.biases_weights[i]))
        if model.use_recurrent:
            num2 = np.sum(model.unit(model.recurrent_w_link))+np.sum(model.unit(model.recurrent_b_link))
        return [num, num2]

    def _get_total_num_links_possible(self, model):
        num = model.num_input * model.layer_nodes[0]
        num2 = -1
        for i in range(1, model.layer_count):
            num += model.layer_nodes[i-1]*model.layer_nodes[i]
        num += model.layer_nodes[model.layer_count-1]*model.num_output
        if model.use_recurrent:
            num2 = model.layer_nodes[model.layer_count-1] * model.layer_nodes[model.layer_count-1] + model.layer_nodes[model.layer_count-1]
        return [num, num2]

    def __initialize_networks(self):
        init_gen = []
        if self._evolve_activation is not None:
            n_act_fun = len(self._evolve_activation)
        if self._evolve_recurrent_layer is not None:
            n_act_fun_rec = len(self._evolve_recurrent_layer)

        for i in range(0, self._max_gen_size):

            if self._evolve_recurrent_layer is not None and self._evolve_activation is not None:
                acts_indices_r = np.random.choice(n_act_fun_rec, 1)
                acts_names_r = []
                for index in acts_indices_r:
                    acts_names_r.append(self._evolve_recurrent_layer[index])
                acts_indices_l = np.random.choice(n_act_fun, len(self.layer_nodes))
                acts_names_l = []
                for index in acts_indices_l:
                    acts_names_l.append(self._evolve_activation[index])
                obj = self.EvolvableNetwork(layer_nodes=self.layer_nodes, num_input=self.num_input,
                                            num_output=self.num_output, use_recurrent=True,
                                            activation_function=acts_names_l,
                                            recurrent_activation=acts_names_r[0],
                                            output_activation=self.default_output_activation,
                                            use_links=self._evolve_links)
            elif self._evolve_recurrent_layer is not None:
                acts_indices = np.random.choice(n_act_fun_rec, 1)
                acts_names = []
                for index in acts_indices:
                    acts_names.append(self._evolve_recurrent_layer[index])
                obj = self.EvolvableNetwork(layer_nodes=self.layer_nodes, num_input=self.num_input,
                                            num_output=self.num_output, use_recurrent=True, recurrent_activation=acts_names[0],
                                            output_activation=self.default_output_activation,
                                            use_links=self._evolve_links)

            elif self._evolve_activation is not None:
                acts_indices = np.random.choice(n_act_fun, len(self.layer_nodes))
                acts_names = []
                for index in acts_indices:
                    acts_names.append(self._evolve_activation[index])
                obj = self.EvolvableNetwork(layer_nodes=self.layer_nodes, num_input=self.num_input,
                                            num_output=self.num_output,
                                            activation_function=acts_names,
                                            output_activation=self.default_output_activation,
                                            use_links=self._evolve_links)
            else:
                obj = self.EvolvableNetwork(layer_nodes=self.layer_nodes, num_input=self.num_input,
                                            num_output=self.num_output,
                                            activation_function=self.default_layer_activation,
                                            output_activation=self.default_output_activation,
                                            use_links=self._evolve_links)
            init_gen.append(obj)
        return init_gen

    def __fitness_function(self, gen):
        output = []
        for i in range(0, len(gen)):
            t = self._error_function(self._expected_output, gen[i].predict(self._input))
            output.append(t)
        return np.asarray(output)

    def __roulette(self, cumsum_fit):
        index = 0
        r = np.random.uniform(0, 1, 1)
        while cumsum_fit[index] < r:
            index += 1
        return index

    def __crossover(self, p1, p2, const_cross):

        child = self.EvolvableNetwork(layer_nodes=p1.layer_nodes, num_input=p1.num_input, num_output=p1.num_output,
                                      activation_function=p1.activation_function_name, output_activation=p1.output_activation_name,
                                      initialize=False, use_links=p1.use_links, use_recurrent=p1.use_recurrent,
                                      recurrent_activation=p1.recurrent_activation_name)

        for i in range(0, p1.layer_count+1):
            child.layers_weights.append((1 - const_cross) * p1.layers_weights[i] + const_cross * p2.layers_weights[i])
            child.biases_weights.append((1 - const_cross) * p1.biases_weights[i] + const_cross * p2.biases_weights[i])
            if self._evolve_links:
                child.layers_links.append((1 - const_cross) * p1.layers_links[i] + const_cross * p2.layers_links[i])
                child.biases_links.append((1 - const_cross) * p1.biases_links[i] + const_cross * p2.biases_links[i])

        if child.use_recurrent:
            child.recurrent_weight = (1 - const_cross) * p1.recurrent_weight + const_cross * p2.recurrent_weight
            child.recurrent_bias = (1 - const_cross) * p1.recurrent_bias + const_cross * p2.recurrent_bias
            if self._evolve_links:
                child.recurrent_w_link = (1 - const_cross) * p1.recurrent_w_link + const_cross * p2.recurrent_w_link
                child.recurrent_b_link = (1 - const_cross) * p1.recurrent_b_link + const_cross * p2.recurrent_b_link

        return child

    def __mutate(self, child, const_mutate):
        for i in range(0, child.layer_count + 1):
            n, c = child.layers_weights[i].shape
            r_w = np.random.uniform(-const_mutate, const_mutate, n*c)
            if self._evolve_links:
                r_l = np.random.uniform(-const_mutate, const_mutate, n * c)
            for nr in range(0, n):
                for nc in range(0, c):
                    child.layers_weights[i][nr, nc] += r_w[nr*c+nc]
                    if self._evolve_links:
                        child.layers_links[i][nr, nc] += r_l[nr * c + nc]
        if child.use_recurrent:
            n, c = child.recurrent_weight.shape
            r_w = np.random.uniform(-const_mutate, const_mutate, n * c)
            if self._evolve_links:
                r_l = np.random.uniform(-const_mutate, const_mutate, n * c)
            for nr in range(0, n):
                for nc in range(0, c):
                    child.recurrent_weight[nr, nc] += r_w[nr * c + nc]
                    if self._evolve_links:
                        child.recurrent_w_link[nr, nc] += r_l[nr * c + nc]
        for i in range(0, child.layer_count + 1):
            c = child.biases_weights[i].shape
            r_w = np.random.uniform(-const_mutate, const_mutate, c[0])
            if self._evolve_links:
                r_l = np.random.uniform(-const_mutate, const_mutate, c[0])
            for nc in range(0, c[0]):
                child.biases_weights[i][nc] += r_w[nc]
                if self._evolve_links:
                    child.biases_links[i][nc] += r_l[nc]
        if child.use_recurrent:
            c = child.recurrent_bias.shape
            r_w = np.random.uniform(-const_mutate, const_mutate, c[0])
            if self._evolve_links:
                r_l = np.random.uniform(-const_mutate, const_mutate, c)
            for nc in range(0, c[0]):
                child.recurrent_bias[nc] += r_w[nc]
                if self._evolve_links:
                    child.recurrent_b_link[nc] += r_l[nc]


    def __reproduce(self, p1, p2):

        amount = 4
        c_cross = np.random.normal(0.5, 0.15, amount)
        children = [p1, p2]
        for i in range(0, amount):
            ch = self.__crossover(p1, p2, c_cross[i])
            if i % 2 == 0:  # mutate only half of offspring
                self.__mutate(ch, 0.001)
            children.append(ch)

        fit = self.__fitness_function(children).tolist()
        if self._error_function == 'r^2':
            del children[np.argmin(fit)]
            del fit[np.argmin(fit)]
            del children[np.argmin(fit)]
            del fit[np.argmin(fit)]
            del children[np.argmin(fit)]
            del fit[np.argmin(fit)]
            del children[np.argmin(fit)]
            del fit[np.argmin(fit)]
        else:
            del children[np.argmax(fit)]
            del fit[np.argmax(fit)]
            del children[np.argmax(fit)]
            del fit[np.argmax(fit)]
            del children[np.argmax(fit)]
            del fit[np.argmax(fit)]
            del children[np.argmax(fit)]
            del fit[np.argmax(fit)]
        #fit = self.__fitness_function(children).tolist()
        return children

    def _create_species_names(self, layer_count, activations, s, species):
        if layer_count > 0:
            for i in range(0, len(activations)):
                k = s + "," + activations[i]
                self._create_species_names(layer_count-1, activations, k, species)
        else:
            species.append(s)

    def get_score(self, avg=False):

        if avg:
            msg = [self._avg_r_2, self._avg_mae, self._avg_mse]
        else:
            msg = [self._best_r_2, self._best_mae, self._best_mse]

        if self._evolve_links:
            if avg:
                msg.append(self._avg_links_used)
            else:
                msg.append(self._best_links_used)
            msg.append(self._total_possible_links)

        msg.append(self.best_model.activation_function_name)
        return msg


    def evolve(self, input, expected_output, error_function='mse', max_generations=100, max_gen_size=100,
               use_previous=False, info=True, tol=1e-10, evolve_links=False, evolve_activation_layer=None,
               evolve_recurrent_layer=None):

        self._input = input
        self._expected_output = expected_output
        self._max_gen_size = max_gen_size
        self._evolve_links = evolve_links
        self._evolve_activation = evolve_activation_layer
        self._evolve_recurrent_layer = evolve_recurrent_layer

        if error_function == 'mse':
            self._error_function = self._mse
        elif error_function == 'mae':
            self._error_function = self._mae
        elif error_function == 'r^2':
            self._error_function = self._r_2

        if evolve_activation_layer is not None or evolve_recurrent_layer is not None:
            species = {}
            id = 0
            species_names = []
            if evolve_recurrent_layer is not None and evolve_activation_layer is None:  # just recurrent
                for spec in evolve_recurrent_layer:
                    species[spec] = id
                    id += 1
            elif evolve_recurrent_layer is not None:  # recurrent and layer
                for spec in evolve_recurrent_layer:
                    self._create_species_names(len(self.layer_nodes), evolve_activation_layer, spec, species_names)
                for s in species_names:
                    species[s] = id
                    id += 1
            elif evolve_activation_layer:  # just layer
                self._create_species_names(len(self.layer_nodes), evolve_activation_layer, "", species_names)
                for s in species_names:
                    species[s[1:]] = id
                    id += 1
            #print(species)

        converged = 0
        if use_previous:
            pass
        else:
            init_gen = self.__initialize_networks()
            start_time = time.time()
            best_fit = []
            mean_fit = []
            prev_best = -1000
            gen = init_gen
            prev_gen_upgrade = 0
            elitism = 0.1
            n = len(gen)
            for k in range(0, max_generations):
                if info:
                    msg = " --- GENERATION {} ---\n".format(k)
                    print(msg)
                total = np.empty(shape=(len(gen),5), dtype=object)
                total[:, 0] = gen
                fit = self.__fitness_function(gen)
                scaled_fit = abs((fit - np.max(fit)) / (np.min(fit) - np.max(fit)))
                if error_function == 'r^2':
                    scaled_fit = abs((fit - np.min(fit)) / (np.max(fit) - np.min(fit)))

                fit = scaled_fit / np.sum(scaled_fit)
                total[:, 1] = fit
                total[:, 3] = range(0, len(gen), 1)
                total = total[np.argsort(total[:,1])]
                total[:, 2] = np.cumsum(total[:, 1])

                elite_index = len(gen) - int(n * elitism)
                elite_include = copy.deepcopy(total[elite_index:len(gen), 0])
                if evolve_activation_layer is not None or evolve_recurrent_layer is not None:
                    if evolve_recurrent_layer is not None and evolve_activation_layer is None:  # just recurrent
                        for i in range(0, len(gen)):
                            s = total[i, 0].recurrent_activation_name
                            total[i, 4] = species[s]
                        spec_present = np.unique(total[:, 4])
                        num_species = len(spec_present)
                        if info:
                            best_species = total[:, 4][-1]
                            for key, value in species.items():
                                if value == best_species:
                                    name = key
                                    break
                            msg = " NUMBER SPECIES PRESENT: {}\n" \
                                  " BEST SPECIES Recurrent: [{}]: \n" \
                                  " TOTAL GEN SIZE: {}".format(len(spec_present), name, len(gen))
                            print(msg)
                    elif evolve_recurrent_layer is not None:  # recurrent and layer
                        for i in range(0, len(gen)):
                            model = total[i, 0].activation_function_name
                            s = total[i, 0].recurrent_activation_name
                            for act in model:
                                s = s + "," + act
                            total[i, 4] = species[s]
                        spec_present = np.unique(total[:, 4])
                        num_species = len(spec_present)
                        if info:
                            best_species = total[:, 4][-1]
                            for key, value in species.items():
                                if value == best_species:
                                    name = key
                                    break
                            msg = " NUMBER SPECIES PRESENT: {}\n" \
                                  " BEST SPECIES: [{}] with RECURRENT [{}]: \n" \
                                  " TOTAL GEN SIZE: {}".format(len(spec_present), name[name.find(',')+1:],name[0:name.find(',')], len(gen))
                            print(msg)
                    else:  # just layer
                        for i in range(0, len(gen)):
                            model = total[i,0].activation_function_name
                            s = ""
                            for act in model:
                                s = s + "," + act
                            total[i,4] = species[s[1:]]
                            #
                        spec_present = np.unique(total[:,4])
                        num_species = len(spec_present)
                        if info:
                            best_species = total[:, 4][-1]
                            for key, value in species.items():
                                if value == best_species:
                                    name = key
                                    break
                            msg = " NUMBER SPECIES PRESENT: {}\n"\
                                  " BEST SPECIES: Recurent: {}\n" \
                                  " TOTAL GEN SIZE: {}".format(len(spec_present), name, len(gen))
                            print(msg)
                    all_children = []
                    keys = list(species.keys())
                    index = 0
                    for spec in spec_present:
                        count = np.count_nonzero(total[:, 4] == spec)
                        if info:
                            if evolve_recurrent_layer is not None and evolve_activation_layer is not None:  # layer and recurrent
                                name = keys[spec]
                                print(" Species [{}] with Recurrent [{}], with count: {}".format(name[name.find(',') + 1:],name[0:name.find(',')] , count))
                            elif evolve_recurrent_layer is not None: # just recurrent
                                name = keys[spec]
                                print(" Species with Recurrent [{}], with count: {}".format(name, count))
                            else:  # just layer
                                print(" Species [{}] with count: {}".format(keys[spec], count))
                        index += 1
                        select = np.empty(shape=(count,), dtype=int)
                        t = (total[:,4] == spec)*(total[:,3]-1)+1
                        ind = t[t != 1].tolist()
                        temp_cum_sum = np.cumsum(fit[ind] / sum(fit[ind]))
                        if count != 1:
                            for i in range(0, count):
                                select[i] = ind[self.__roulette(temp_cum_sum)]
                        else:
                            select[0] = ind[0]
                        mates = np.random.choice(select, count, replace=False)
                        for i in range(0, count):
                            children = self.__reproduce(gen[select[i]], gen[mates[i]])
                            for child in children:
                                all_children.append(child)
                    if info:
                        print("")
                else:
                    total[:, 4] = range(0, n, 1)


                    selection = np.empty(shape=(n,), dtype=int)
                    for i in range(0, n):
                        selection[i] = total[self.__roulette(total[:, 2]), 3]
                    mates = np.random.choice(selection, n, replace=False)
                    all_children = []
                    for i in range(0, n):
                        children = self.__reproduce(gen[selection[i]], gen[mates[i]])
                        for child in children:
                            all_children.append(child)


                if evolve_activation_layer is None and evolve_recurrent_layer is None:  # neither
                    gen_next = all_children
                    n2 = len(gen_next)
                    fit_next = self.__fitness_function(gen_next)
                    if error_function == 'r^2':
                        s_min = np.min(fit_next)
                        s_max = np.max(fit_next)
                        fit_next = (fit_next - s_min) / (s_max - s_min)
                    total = np.empty(shape=(n2, 3), dtype=object)
                    total[:, 0] = gen_next
                    total[:, 1] = np.array(fit_next, copy=True)
                    if error_function == 'r^2':
                        total = total[np.argsort(-total[:, 1])]  # want to find max R^2
                    else:
                        total = total[np.argsort(total[:, 1])]  # want to find min mae/mse
                    total[:, 2] = range(0, n2)
                    indices_replace = np.array(total[n:n2, 2], dtype=int).tolist()
                    total = np.delete(total, indices_replace, axis=0)


                    total[elite_index:n, 0] = elite_include
                    total[elite_index:n, 1] = self.__fitness_function(elite_include)
                else:
                    gen_next = all_children
                    n2 = len(gen_next)
                    fit_next = self.__fitness_function(gen_next)
                    if error_function == 'r^2':
                        s_min = np.min(fit_next)
                        s_max = np.max(fit_next)
                        fit_next = (fit_next - s_min) / (s_max - s_min)
                    total = np.empty(shape=(n2, 4), dtype=object)
                    total[:, 0] = gen_next
                    total[:, 1] = np.array(fit_next, copy=True)
                    if error_function == 'r^2':
                        total = total[np.argsort(-total[:, 1])]  # want to find max R^2
                    else:
                        total = total[np.argsort(total[:, 1])]  # want to find min mae/mse
                    total[:, 2] = range(0, n2)

                    elite_index = n2 - len(elite_include)
                    # print(self.__fitness_function(elite_include)[-1])
                    total[elite_index:n2, 0] = elite_include
                    total[elite_index:n2, 1] = self.__fitness_function(elite_include)
                    if error_function == 'r^2':
                        total = total[np.argsort(-total[:, 1])]  # want to find max R^2
                    else:
                        total = total[np.argsort(total[:, 1])]  # want to find min mae/mse
                    if k >= 15 and evolve_recurrent_layer is None:
                        indices_replace = np.array(total[n:n2, 2], dtype=int).tolist()
                        total = np.delete(total, indices_replace, axis=0)
                    elif k >= 8:
                        indices_replace = np.array(total[n:n2, 2], dtype=int).tolist()
                        total = np.delete(total, indices_replace, axis=0)
                    else:
                        total[:, 2] = range(0, n2)
                        if evolve_recurrent_layer is not None and evolve_activation_layer is not None:  # layer and recurrent
                            for i in range(0, n2):
                                model = total[i, 0].activation_function_name
                                s = total[i, 0].recurrent_activation_name
                                for act in model:
                                    s = s + "," + act
                                total[i, 3] = species[s]
                        elif evolve_recurrent_layer is not None:  # just recurrent
                            for i in range(0, n2):
                                s = total[i, 0].recurrent_activation_name
                                total[i, 3] = species[s]
                        else:  # just layer
                            for i in range(0, n2):
                                model = total[i, 0].activation_function_name
                                s = ""
                                for act in model:
                                    s = s + "," + act
                                total[i, 3] = species[s[1:]]
                        spec_present = np.unique(total[:, 3])
                        index = 0
                        all_ind = []
                        for spec in spec_present:
                            count = np.count_nonzero(total[:, 3] == spec)
                            if count <= 5:
                                continue
                            index += 1
                            t = (total[:, 3] == spec) * (total[:, 2]-1)+1
                            ind = t[t != 1].tolist()
                            if evolve_recurrent_layer is not None and evolve_activation_layer is not None or evolve_activation_layer:
                                if count > 1000:
                                    ind = ind[int(len(ind)*0.01):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                elif count > 500:
                                    ind = ind[int(len(ind)*0.1):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                elif count > 100:
                                    ind = ind[int(len(ind)*0.2):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                elif count > 50:
                                    ind = ind[int(len(ind) * 0.3):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                elif count > 20:
                                    ind = ind[int(len(ind) * 0.4):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                elif count > 10:
                                    ind = ind[int(len(ind) * 0.5):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                else:
                                    ind = ind[int(len(ind) * 0.8):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                            elif evolve_recurrent_layer is not None:
                                if count > 400:
                                    ind = ind[int(len(ind) * 0.5):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                elif count > 100:
                                    ind = ind[int(len(ind) * 0.6):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                elif count > 50:
                                    ind = ind[int(len(ind) * 0.7):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                elif count > 20:
                                    ind = ind[int(len(ind) * 0.8):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                                else:
                                    ind = ind[int(len(ind) * 0.9):len(ind)]
                                    if ind:
                                        all_ind.append(ind)
                            else:
                                pass
                        del_ind = list(itertools.chain.from_iterable(all_ind))
                        total = np.delete(total, del_ind, axis=0)
                gen_next = total[:, 0]
                if error_function == 'r^2':
                    total[:, 1] = (s_max-s_min)*total[:, 1]+s_min

                fit_mean = np.mean(total[:, 1])
                fit_best = np.min(total[:, 1])
                if info:
                    msg = " Best {}: {}\n" \
                          " Mean {}: {}\n".format(error_function, fit_best, error_function, fit_mean)
                    print(msg)
                mean_fit.append(fit_mean)
                best_fit.append(fit_best)
                gen = gen_next
                if abs(prev_best - fit_best) > tol:
                    prev_gen_upgrade = k
                elif k-prev_gen_upgrade == 25:
                    if info:
                        msg = " -- BEST FIT HAS NOT INCREASED FOR 25 GENERATIONS - STOPPING \n"
                        print(msg)
                    finish_time = time.time()
                    converged = 1
                    break
                if k == max_generations - 1:
                    if info:
                        msg = " MAXIMUM ITERATIONS REACHED WITH DIFFERENCE: {}\n".format(abs(prev_best-fit_best))
                        print(msg)
                prev_best = fit_best

            if error_function == 'r^2':
                total = total[np.argsort(-total[:, 1])]  # want to find max R^2
            else:
                total = total[np.argsort(total[:, 1])]  # want to find min mae/mse
            self.best_model = total[0, 0]
            self.best_3 = [self.best_model]
            self.best_3_fit = [total[0, 1]]
            for i in range(1, int(max_gen_size)):
                if total[i, 1] not in self.best_3_fit:
                    self.best_3.append(total[i, 0])
                    self.best_3_fit.append(total[i, 1])
                if len(self.best_3) == 3:
                    break

            if not converged:
                finish_time = time.time()
            elapsed = finish_time - start_time
            if info:
                msg = " TOTAL TIME TAKEN: {}\n".format(elapsed)
                print(msg)

            if info:
                og_error = self._error_function
                self._error_function = self._r_2
                r_2 = self.__fitness_function([self.best_model])
                r_2_3 = np.mean(self.__fitness_function(self.best_3))
                self._error_function = self._mse
                mse = self.__fitness_function([self.best_model])
                mse_3 = np.mean(self.__fitness_function(self.best_3))
                self._error_function = self._mae
                mae = self.__fitness_function([self.best_model])
                mae_3 = np.mean(self.__fitness_function(self.best_3))
                self._error_function = og_error
                self._best_r_2 = r_2[0]
                self._best_mae = mae[0]
                self._best_mse = mse[0]
                self._avg_r_2 = r_2_3
                self._avg_mae = mae_3
                self._avg_mse = mse_3
                if evolve_links:
                    links_used = self._get_num_links_used(self.best_model)
                    if len(self.best_3) == 3:
                        t1 = self._get_num_links_used(self.best_3[0])
                        t2 = self._get_num_links_used(self.best_3[1])
                        t3 = self._get_num_links_used(self.best_3[2])
                        t = [(t1[0] + t2[0] + t3[0])/3, (t1[1] + t2[1] + t3[1])/3]
                        avg_links_used = t
                    else:
                        avg_links_used = self._get_num_links_used(self.best)
                    total_possible_links = self._get_total_num_links_possible(self.best_model)
                    self._best_links_used = links_used
                    self._total_possible_links = total_possible_links
                    self._avg_links_used = avg_links_used
                    msg = " -- BEST MODEL SCORE -- \n" \
                          " R^2: {}\n" \
                          " MAE: {}\n" \
                          " MSE: {}\n" \
                          " TOTAL LINKS USED: {} OUT OF {}\n\n" \
                          " -- AVG BEST THREE MODEL SCORE -- \n" \
                          " R^2: {}\n" \
                          " MAE: {}\n" \
                          " MSE: {}\n" \
                          "AVG LINKS USED: {} OUT OF {}\n".format(r_2[0], mae[0], mse[0], links_used,
                                                                  total_possible_links, r_2_3, mae_3, mse_3,avg_links_used,
                                                                  total_possible_links)
                else:
                    msg = " -- BEST MODEL SCORE -- \n" \
                          " R^2: {}\n" \
                          " MAE: {}\n" \
                          " MSE: {}\n\n" \
                          " -- AVG BEST THREE MODEL SCORE -- \n" \
                          " R^2: {}\n" \
                          " MAE: {}\n" \
                          " MSE: {}\n".format(r_2[0], mae[0], mse[0], r_2_3, mae_3, mse_3)

                print(msg)

    def predict(self, input, avg=False):
        if avg:
            out = 0
            for model in self.best_3:
                out += model.predict(input)
            out = out / 3
        else:
            out = self.best_model.predict(input)

        return out

    def track(self):
        pass
