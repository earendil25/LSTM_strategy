class LSTM_module():
    def __init__(self, feature_label_df, feature_config):

        self.feature_label_df = feature_label_df
        self.feature_config = feature_config

        self.total_kde_dic = {}
        self.up_kde_dic = {}
        self.prob_up_dic = {}

    def get_train_data(self, train_start, train_end):

        train_start = dt.datetime.strptime(train_start, '%Y-%m-%d')
        train_end = dt.datetime.strptime(train_end, '%Y-%m-%d')

        feature_label_df = self.feature_label_df
        feature_date_list = list(feature_label_df.index)

        while True:
            try:
                idx_start = feature_date_list.index(train_start)
                break
            except:
                train_start += dt.timedelta(days = 1)

        while True:
            try:
                idx_end = feature_date_list.index(train_end) + 1
                break
            except:
                train_end -= dt.timedelta(days = 1)

        print(train_start, train_end)

        train_date_list = feature_date_list[idx_start:idx_end]

        self.feature_config['train_date'] = [train_date_list[0], train_date_list[-1]]

        ipt_to_train_feature_array = {
            ipt : np.array([feature_label_df[ipt].loc[date] for date in train_date_list])
            for ipt in self.feature_config['inputs']}

        train_label_array = np.array([feature_label_df.loc[date]['label'] for date in train_date_list ])

        self.ipt_to_train_feature_array = ipt_to_train_feature_array
        self.train_label_array = train_label_array

    def set_search_space(self, search_space):
        self.search_space = search_space

    def set_model_config(self, config):
        self.model_config = config

    def initialize_model(self, hp, is_tuning = True):

        loss_type = self.model_config['loss_type']

        if is_tuning:
            is_batchnormalize_list = self.search_space['is_batchnormalize']
            LSTM_node_size_list = self.search_space['LSTM_node_size']
            dense_node_size_list = self.search_space['dense_node_size']
            LSTM_depth_list = self.search_space['LSTM_depth']
            dense_depth_list = self.search_space['dense_depth']
            dropout_list = self.search_space['dropout']
            log_lr_list = self.search_space['log_lr']
            is_resnet_list = self.search_space['is_resnet']

            is_batchnormalize = hp.Choice('is_batchnormalize', values = is_batchnormalize_list)
            is_resnet = hp.Choice('is_resnet', values = is_resnet_list)
            LSTM_node_size = hp.Choice('node_size', values = LSTM_node_size_list)
            dense_node_size = hp.Choice('node_size', values = dense_node_size_list)
            LSTM_depth = hp.Choice('LSTM_depth', values = LSTM_depth_list)
            dense_depth = hp.Choice('dense_depth', values = dense_depth_list)

            if len(dropout_list) == 3:
                dropout = hp.Float('dropout', 
                                min_value = dropout_list[0], 
                                max_value = dropout_list[1], 
                                step = dropout_list[2])

            elif len(dropout_list) == 1:
                dropout = hp.Fixed('dropout', value = dropout_list[0])

            if len(log_lr_list) == 3:
                log_lr = hp.Float('log_lr', 
                                min_value = log_lr_list[0], 
                                max_value = log_lr_list[1], 
                                step = log_lr_list[2])

            elif len(log_lr_list) == 1:
                log_lr = hp.Fixed('log_lr', value = log_lr_list[0])

        else:
            LSTM_node_size = self.model_config['LSTM_node_size']
            dense_node_size = self.model_config['dense_node_size']
            LSTM_depth = self.model_config['LSTM_depth']
            dense_depth = self.model_config['dense_depth']
            dropout = self.model_config['dropout']
            log_lr = self.model_config['log_lr']
            is_batchnormalize = self.model_config['is_batchnormalize']
            is_resnet = self.model_config['is_resnet']

        lr = 10.**log_lr

        ########################################################################
        initializer = tf.keras.initializers.HeNormal(seed = 1)

        def LSTM(x):
            return tf.keras.layers.LSTM(LSTM_node_size, 
                                    activation = 'tanh',
                                    recurrent_activation  = 'sigmoid',
                                    return_sequences = True,
                                    recurrent_dropout = 0.0,
                                    kernel_initializer = initializer)(x)

        def LSTM_false(x):
            return tf.keras.layers.LSTM(LSTM_node_size, 
                                    activation = 'tanh',
                                    recurrent_activation  = 'sigmoid',
                                    return_sequences = False,
                                    recurrent_dropout = 0.0,
                                    kernel_initializer = initializer)(x)

        def FCL(x, size = dense_node_size):
            return tf.keras.layers.Dense(size, 
                                    activation='linear',
                                    kernel_initializer = initializer)(x)

        def ReLU(x):
            return tf.keras.layers.ReLU()(x)

        def Sigmoid(x):
            return tf.keras.layers.Activation('sigmoid')(x)

        def Dropout(x):
            return tf.keras.layers.Dropout(dropout)(x)

        def BN(x):
            return tf.keras.layers.BatchNormalization()(x)

        def LN(x):
            return tf.keras.layers.LayerNormalization()(x)

        def res_block(x, size = dense_node_size):
            x = Dropout(x)
            x = ReLU(x)
            x = FCL(x, size)
            
    
            return x

        ########################################################################

        input_layer = {ipt : tf.keras.Input(
                        shape = self.ipt_to_train_feature_array[ipt].shape[1:])
                        for ipt in self.feature_config['inputs']}

        ########################################################################

        LSTM_layer = {}
        for ipt in self.feature_config['inputs']:
            layer = input_layer[ipt]

            for _ in range(LSTM_depth):
                layer = LSTM(layer)
                
            layer = tf.keras.layers.Flatten()(layer)

            layer = FCL(layer)

            res_layer = layer
            for _ in range(is_resnet):
                res_layer = res_block(res_layer)
            
            layer = layer + res_layer
            layer = Dropout(layer)

            layer  = ReLU(layer)
            LSTM_layer[ipt] = layer

        layer = tf.keras.layers.concatenate([LSTM_layer[ipt] for ipt in self.feature_config['inputs']])
    
        ########################################################################

        layer = FCL(layer)

        for x in range(dense_depth):
            in_layer = layer

            res_layer = layer
            for _ in range(is_resnet):
                res_layer = res_block(res_layer)

            layer = in_layer + res_layer
            layer = Dropout(layer)
        
        layer = ReLU(layer)

        ########################################################################

        layer = FCL(layer, size = len(self.feature_config['universe_list']))

        res_layer = layer
        for _ in range(is_resnet-1):
            res_layer = res_block(res_layer)
        res_layer = res_block(res_layer, size = len(self.feature_config['universe_list']))

        layer = layer + res_layer

        out = Sigmoid(layer)
        
        ########################################################################

        self.model = tf.keras.Model(inputs=[input_layer[ipt] for ipt in self.feature_config['inputs']], 
                                    outputs=out)

        self.model.compile(loss = loss_type, 
                           optimizer = tf.keras.optimizers.Adam(lr = lr),
                           metrics = ['accuracy'])

        return self.model
        
    def train_model(self, epochs = 10, patience = 1, is_tensorboard = False, validation_split = 0.2):

        batch_size = self.model_config['batch_size']

        filepath = os.path.join('tmp_checkpoint.h5')

        RoP_callback = tf.keras.callbacks.ReduceLROnPlateau(
                                                            monitor="val_loss",
                                                            factor=10**(-0.2),
                                                            patience=2,
                                                            verbose=1,
                                                            mode="auto",
                                                            min_delta=0.0,
                                                            cooldown=0,
                                                            min_lr=0
                                                        )


        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience = patience,
                                                    restore_best_weights = True)
        
        check_point = tf.keras.callbacks.ModelCheckpoint(
                        filepath = filepath, monitor='val_loss', verbose=1, save_best_only=True,
                        save_weights_only=True, mode='auto', save_freq='epoch',
                        options=None )
        
        terminateOnNAN = tf.keras.callbacks.TerminateOnNaN()

        csv_logger = tf.keras.callbacks.CSVLogger('log.csv', append=True, separator=';')

        callbacks = [RoP_callback, early_stopping, check_point, csv_logger, terminateOnNAN]

        
        if is_tensorboard:
            log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks += [tensorboard_callback]

        history = self.model.fit(
            [self.ipt_to_train_feature_array[ipt] for ipt in self.feature_config['inputs']], 
            self.train_label_array, 
            epochs = epochs,
            validation_split = validation_split,
            verbose = 1,
            batch_size = batch_size, 
            shuffle = True,
            callbacks= callbacks)

        self.model.load_weights(filepath)

        return history

    def get_prediction(self, test_start, test_end, zero_padding = {}):

        test_start = dt.datetime.strptime(test_start, '%Y-%m-%d')
        test_end = dt.datetime.strptime(test_end, '%Y-%m-%d')

        feature_date_list = list(self.feature_label_df.index)

        while True:
            try:
                idx_start = feature_date_list.index(test_start)
                break
            except:
                test_start += dt.timedelta(days = 1)

        while True:
            try:
                idx_end = feature_date_list.index(test_end) + 1
                break
            except:
                test_end -= dt.timedelta(days = 1)

        print(test_start, test_end)

        date_list = feature_date_list[idx_start:idx_end]

        ipt_to_zero_padding = {
            ipt : [feature not in zero_padding[ipt]
                    for feature in self.feature_config['inputs'][ipt]['feature_list']]
            for ipt in self.feature_config['inputs']
        }


        ipt_to_feature = {
            ipt: np.array([[item * ipt_to_zero_padding[ipt]
                        for item in self.feature_label_df.loc[date][ipt]]
                        for date in date_list])
            for ipt in self.feature_config['inputs']
            }


        input_data = [ipt_to_feature[ipt] for ipt in self.feature_config['inputs']]

        prediction_list = list(self.model.predict(input_data))

        label_list = [self.feature_label_df.loc[date]['label'] for date in date_list]

        return pd.DataFrame({'prediction':prediction_list,
                            'label' : label_list},
                            index = date_list)

    def get_KDE(self, test_start, test_end, zero_padding = {}):
        compute_df = self.get_prediction(test_start, test_end, zero_padding)

        total_kde_dic, up_kde_dic, prob_up_dic = {}, {}, {}

        def _get_kde(target_array):
            target_array = target_array.reshape(-1,1)
            kde = KernelDensity(kernel='gaussian', bandwidth = 0.1).fit(target_array)
            return kde

        for idx in range(len(compute_df['prediction'][0])):
            test_set = [ [compute_df['prediction'].loc[x][idx], compute_df['label'].loc[x][idx] ] 
                        for x in compute_df.index]
            total_set, up_set = [], []
            for elem in test_set:
                total_set.append(elem[0])
                if elem[1] == 1: up_set.append(elem[0])

            total_array = np.array(total_set)
            up_array = np.array(up_set)
            
            total_kde_dic[idx] = _get_kde(total_array)
            up_kde_dic[idx] = _get_kde(up_array)
            prob_up_dic[idx] = len(up_set)/len(total_set)

        self.total_kde_dic = total_kde_dic
        self.up_kde_dic = up_kde_dic
        self.prob_up_dic = prob_up_dic

    def prob_converter(self, value_list):

        def _get_prob(value, kde):
            return np.exp(kde.score_samples(np.array([value]).reshape(-1,1))[0])

        pp_list = []

        for idx in range(len(value_list)):
            value = value_list[idx]
            total_kde = self.total_kde_dic[idx]
            up_kde = self.up_kde_dic[idx]
            prob_up = self.prob_up_dic[idx]
            pp = prob_up * _get_prob(value, up_kde) / _get_prob(value, total_kde)
            pp_list.append(pp)
            
        return pp_list