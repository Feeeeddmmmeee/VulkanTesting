struct VSInput {
	float3 inPos;
	float3 inColor;
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
};

[shader("vertex")]
VSOutput vertMain(VSInput input)
{
	VSOutput out;
	out.pos = mul(ubo.proj, mul(ubo.view ,mul(ubo.model, float4(input.inPos, 1.0))));
	out.color = input.inColor;
	return out;
}

[shader("fragment")]
float4 fragMain(VSOutput inVert) : SV_Target
{
	return float4(inVert.color, 1);
}
