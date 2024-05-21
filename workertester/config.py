from pydantic import BaseModel, Field
from typing import Optional

class EndpointConfig(BaseModel):
    name: str = Field(..., description="The name of the endpoint")
    template_id: str = Field(..., description="The id of the template to use for the endpoint")
    gpu_ids: str = Field("AMPERE_16", choices= ["AMPERE_16",  "AMPERE_24","ADA_24", 
                                                "AMPERE_48", "ADA_48_PRO",
                                                "AMPERE_80", "ADA_80_PRO"], 
                         description="The gpu ids to use for the endpoint")
    network_volume_id: Optional[str] = Field(None, description="The id of the network volume to use for the endpoint")
    locations: Optional[str] = Field(None, description="The locations to use for the endpoint")
    idle_timeout: int = Field(5, description="The idle timeout for the endpoint")
    scaler_type: str = Field("QUEUE_DELAY", description="The scaler type for the endpoint")
    scaler_value: int = Field(4, description="The scaler value for the endpoint")
    workers_min: int = Field(0, description="The minimum number of workers for the endpoint")
    workers_max: int = Field(3, description="The maximum number of workers for the endpoint")
    flashboot: bool = Field(False, description="Whether to enable flashboot for the endpoint")
    
class vLLMWorkerConfig(EndpointConfig):
    flashboot: bool = True
    workers_min: int = 1
    workers_max: int = 1
    