import tensorflow as tf
from tensorflow.keras import layers

# class generate_patch(layers.Layer):
#     def __init__(self, patch_size):
#         super(generate_patch, self).__init__()
#         self.patch_size = patch_size

#     def call(self, images):
#         batch_size = tf.shape(images)[0]
#         patches = tf.image.extract_patches(images=images,
#                                            sizes=[1, self.patch_size, self.patch_size, 1],
#                                            strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1],
#                                            padding="VALID")
#         patch_dims = patches.shape[-1]
#         patches = tf.reshape(patches, [batch_size, -1,
#                                        patch_dims])  # here shape is (batch_size, num_patches, patch_h*patch_w*c)
#         return patches

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'patch_size': self.patch_size,
#         })
#         return config


# ### Positonal Encoding Layer
# class PatchEncode_Embed(layers.Layer):
#     '''
#     2 steps happen here
#         1. flatten the patches
#         2. Map to dim D; patch embeddings
#     '''

#     def __init__(self, num_patches, projection_dim):
#         super(PatchEncode_Embed, self).__init__()
#         self.num_patches = num_patches
#         self.projection_dim = projection_dim
#         self.projection = layers.Dense(units=self.projection_dim)
#         self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

#     def call(self, patch):
#         positions = tf.range(start=0, limit=self.num_patches, delta=1)
#         encoded = self.projection(patch) + self.position_embedding(positions)
#         return encoded

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'num_patches': self.num_patches,
#             'projection_dim': self.projection_dim,
#         })
#         return config

# Generate Patch Convulation
def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
  patches = layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
  row_axis, col_axis = (1, 2) # channels last images
  seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
  x = tf.reshape(patches, [-1, seq_len, hidden_size])
  return x

# Add Position Embeddings
class AddPositionEmbs(layers.Layer):
  """inputs are image patches
  Custom layer to add positional embeddings to the inputs."""

  def __init__(self, posemb_init=None, **kwargs):
    super().__init__(**kwargs)
    self.posemb_init = posemb_init
    #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

    return inputs + pos_embedding

  def get_config(self):
      config = super().get_config().copy()
      config.update({
          'posemb_init': self.posemb_init,
      })
      return config

pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))

# Position Embedding Layer
# pel = APE(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name="Temp")

# Multi Layer Perceptron Block
def mlp_block_f(mlp_dim, inputs):
    x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
    x = layers.Dropout(rate=0.1)(x)  # dropout rate is from original paper,
    x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x)  # check GELU paper
    x = layers.Dropout(rate=0.1)(x)
    return x


def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
    x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x)
    # self attention multi-head, dropout_rate is from original implementation
    x = layers.Add()([x, inputs])  # 1st residual part

    y = layers.LayerNormalization(dtype=x.dtype)(x)
    y = mlp_block_f(mlp_dim, y)
    y_1 = layers.Add()([y, x])  # 2nd residual part
    return y_1


def Encoder_f(num_layers, mlp_dim, num_heads, inputs, name):
    x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name=name)(inputs)
    x = layers.Dropout(rate=0.2)(x)
    for _ in range(num_layers):
        x = Encoder1Dblock_f(num_heads, mlp_dim, x)

    encoded = layers.LayerNormalization(name=f'{name}-encoder_norm')(x)
    return encoded

transformer_layers = 6
patch_size = 4
hidden_size = 64
num_heads = 4
mlp_dim = 128
img_sizes = ((256, 256, 3), (128, 128, 1))

def build_DRAN(img_sizes, patch_size, hidden_size, num_heads, mlp_dim):
    inputsSegmentor = layers.Input(shape=img_sizes[0])
    inputsRadar = layers.Input(shape=img_sizes[1])

    rescaleSegmentor = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])(inputsSegmentor)
    rescaleRadar = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1. / 255)])(inputsRadar)

    patchesSegmentor = generate_patch_conv_orgPaper_f(patch_size, hidden_size, rescaleSegmentor)
    patchesRadar = generate_patch_conv_orgPaper_f(patch_size, hidden_size, rescaleRadar)

    print("Patches Made")

    encoderOutSegmentor = Encoder_f(transformer_layers, mlp_dim, num_heads, patchesSegmentor, name="segmentor")
    encoderOutRadar = Encoder_f(transformer_layers, mlp_dim, num_heads, patchesRadar, name="radar")

    print("Encoders made")

    representationSegmentor = layers.LayerNormalization(epsilon=1e-6)(encoderOutSegmentor)
    representationSegmentor = layers.Flatten()(representationSegmentor)
    representationSegmentor = layers.Dropout(0.5)(representationSegmentor)
    representationSegmentor = layers.Dense(1024, activation='relu')(representationSegmentor)

    representationRadar = layers.LayerNormalization(epsilon=1e-6)(encoderOutRadar)
    representationRadar = layers.Flatten()(representationRadar)
    representationRadar = layers.Dropout(0.5)(representationRadar)
    representationRadar = layers.Dense(1024, activation='relu')(representationRadar)

    representation = layers.concatenate([representationRadar, representationSegmentor])

    features = mlp_block_f(mlp_dim, representation)

    print("Features Made")

    throttleVal = layers.Dense(2048, activation='relu')(features)
    throttleVal = layers.Dense(128, activation='relu')(throttleVal)
    throttleVal = layers.Dense(1, activation='sigmoid', name="throttle_value")(throttleVal)

    throttleFlag = layers.Dense(2048, activation='relu')(features)
    throttleFlag = layers.Dense(128, activation='relu')(throttleFlag)
    throttleFlag = layers.Dense(1, activation='sigmoid', name="throttle_flag")(throttleFlag)

    steeringVal = layers.Dense(2048, activation='relu')(features)
    steeringVal = layers.Dense(128, activation='relu')(steeringVal)
    steeringVal = layers.Dense(1, activation='sigmoid', name="steering_value")(steeringVal)

    steeringFlag = layers.Dense(2048, activation='relu')(features)
    steeringFlag = layers.Dense(128, activation='relu')(steeringFlag)
    steeringFlag = layers.Dense(1, activation='sigmoid', name="steering_flag")(steeringFlag)

    print("Model final layers added")

    model = tf.keras.Model(inputs=[inputsSegmentor, inputsRadar], outputs=[throttleVal, throttleFlag, steeringVal, steeringFlag], name="DRAN")

    return model


if __name__ == "__main__":
    print("Started Building Model")
    model = build_DRAN(img_sizes, patch_size, hidden_size, num_heads, mlp_dim)

    tf.keras.utils.plot_model(
        model,
        to_file='../files/plot.png',
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=False,
        show_layer_activations=True
    )
    with open('../files/summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.save("../files/dranNN.h5")