"""Import all the modules."""


from tfold.modules.embedding.channel_position_embedding import ChainRelativePositionEmbedding
from tfold.modules.embedding.multimer_position_embedding import MultimerPositionEmebedding
from tfold.modules.embedding.position_embedding import SinusoidalPositionEmbedding
from tfold.modules.embedding.relative_position_embedding import RelativePositionEmbedding
from tfold.modules.embedding.contact_embedding import ContactEmebedding
from tfold.modules.embedding.ppi_embedding import PPIEmbedding
from tfold.modules.embedding.learnable_embedding import LearnableResidueEmbedding


__all__ = [
    'ChainRelativePositionEmbedding',
    'MultimerPositionEmebedding',
    'SinusoidalPositionEmbedding',
    'RelativePositionEmbedding',
    'ContactEmebedding',
    'PPIEmbedding',
    'LearnableResidueEmbedding',
]
