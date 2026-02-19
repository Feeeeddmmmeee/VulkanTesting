static float2 triangle[3] = float2[](
	float2(0.0, -0.5),
	float2(0.5, 0.5),
	float2(-0.5, 0.5)
);

static float3 colors[3] = float3[](
	float3(1,0,0),
	float3(0,1,0),
	float3(0,0,1)
);

struct VertexOutput {
	float3 color;
	float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vertMain(uint vid : SV_VertexID)
{
	VertexOutput out;
	out.sv_position = float4(triangle[vid], 0.0, 1.0);
	out.color = colors[vid];
	return out;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target
{
	return float4(inVert.color, 1);
}
