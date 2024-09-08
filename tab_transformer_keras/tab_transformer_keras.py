from tensorflow import keras
from tensorflow.keras import layers

# Define TransformerBlock class
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define TabTransformer class
class TabTransformer(keras.Model):
    def __init__(self, 
                 categories,
                 num_continuous,
                 dim,
                 dim_out,
                 depth,
                 heads,
                 attn_dropout,
                 ff_dropout,
                 mlp_hidden,
                 normalize_continuous=True):
        super(TabTransformer, self).__init__()

        # Continuous inputs
        self.normalize_continuous = normalize_continuous
        if normalize_continuous:
            self.continuous_normalization = layers.LayerNormalization()

        # Categorical inputs
        self.embedding_layers = [layers.Embedding(input_dim=num_classes, output_dim=dim) for num_classes in categories]
        self.embedded_concatenation = layers.Concatenate(axis=1)

        # Transformers
        self.transformers = [TransformerBlock(dim, heads, dim, rate=attn_dropout) for _ in range(depth)]
        self.flatten_transformer_output = layers.Flatten()

        # MLP
        self.pre_mlp_concatenation = layers.Concatenate()
        self.mlp_layers = [layers.Dense(size, activation=activation) for size, activation in mlp_hidden]
        self.output_layer = layers.Dense(dim_out)

    def call(self, inputs, training=False):
        continuous_inputs = inputs[0]
        categorical_inputs = inputs[1:]
        
        # Continuous
        if self.normalize_continuous:
            continuous_inputs = self.continuous_normalization(continuous_inputs)

        # Categorical
        embedding_outputs = [embedding_layer(categorical_input) for categorical_input, embedding_layer in zip(categorical_inputs, self.embedding_layers)]
        categorical_inputs = self.embedded_concatenation(embedding_outputs)

        for transformer in self.transformers:
            categorical_inputs = transformer(categorical_inputs, training=training)
        
        contextual_embedding = self.flatten_transformer_output(categorical_inputs)

        # MLP
        mlp_input = self.pre_mlp_concatenation([continuous_inputs, contextual_embedding])
        for mlp_layer in self.mlp_layers:
            mlp_input = mlp_layer(mlp_input)

        return self.output_layer(mlp_input)