class Config(object):
    """
    define a class to store parameters,
    """

    def __init__(self):
        self.name = "AnchorFree"
        self.seed = 1129
        
        self.unit_size = 1
        self.feature_dim = 500
        self.window_size = 768
        self.window_step = 128 
        self.inference_window_step = 256 

        self.num_classes = 2
        self.batch_size = 16

        self.weight_decay = 0.0001
        self.checkpoint_path = "./checkpoint/"
        self.epoch = 15

        self.loss_lamda = 1.0
        self.training_lr = 0.01
        self.lr_scheduler_step = 10
        self.lr_scheduler_gama = 0.1

        
        
        self.nms_pre = 100
        self.score_thresold = 0.05
        self.nms_thresold = 0.5
        self.max_per_video = 10


        # model
        self.vis_dim = 500

        #Language
        self.ntoken = 9091
        self.word_dim = 300
        self.sent_hidden_dim = 250
        self.sent_dim = self.sent_hidden_dim*2

