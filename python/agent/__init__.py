from agent.agent_vqvae import VQVAEAgent
from agent.agent_geovae import GeoVAEAgent
from agent.agent_spvae import SPVAEAgent
from agent.agent_pixelsnail import PixelSNAILAgent
from agent.agent_pixelsnail_others import PixelSNAILOthersAgent

def get_agent(config):
    model_name = config['model']['name']
    if model_name == 'vqvae':
        return VQVAEAgent(config)
    elif model_name == 'geovae':
        return GeoVAEAgent(config)
    elif model_name == 'spvae':
        return SPVAEAgent(config)
    elif model_name == 'pixelsnail_top_center' or \
        model_name == 'pixelsnail_bottom_center':
        return PixelSNAILAgent(config)
    elif model_name == 'pixelsnail_top_others' or \
        model_name == 'pixelsnail_bottom_others':
        return PixelSNAILOthersAgent(config)
    else:
        raise ValueError