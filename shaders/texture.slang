struct VSInput {
	float2 inPos;
	float3 inColor;
	float2 inTexCoord;
};

struct UniformBuffer {
	float4x4 model;
	float4x4 view;
	float4x4 proj;
};
ConstantBuffer<UniformBuffer> ubo;

struct VSOutput {
	float4 pos : SV_Position;
	float3 color;
	float2 texCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
	VSOutput out;
	out.pos = mul(ubo.proj, mul(ubo.view ,mul(ubo.model, float4(input.inPos, 0.0, 1.0))));
	out.color = input.inColor;
	out.texCoord = input.inTexCoord;
	return out;
}

Sampler2D texture;

[shader("fragment")]
float4 fragMain(VSOutput inVert) : SV_Target
{
	return texture.Sample(inVert.texCoord);
}
