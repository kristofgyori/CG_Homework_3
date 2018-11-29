# Boy's surface
``` cpp
//---------------------------
class BoySurface : public ParamSurface {
//---------------------------
private:
	float R, w;

	vec3 drdV(float u, float v) {
		float f_x = (sqrtf(2.0)*pow(cosf(v), 2)*cosf(2 * u) + cosf(u)*sinf(2 * v)); // Számláló
		float g_x = (2 - sqrtf(2.0)*sinf(3 * u)*sinf(2 * v));                         // Nevező
		float df_x = (2 * sqrtf(2.0)*cosf(2 * u)*cosf(v)*(-sinf(v))) + (cosf(u)*cosf(2 * v) * 2);
		float dg_x = -sqrtf(2.0)*sinf(3 * u)*cosf(2*v) * 2;

		float f_y = (sqrtf(2.0)*pow(cosf(v), 2)*cosf(2 * u) + cosf(u)*sinf(2 * v)); // Számláló
		float g_y = g_x;                                                        // Nevező
		float df_y = (2 * sqrtf(2.0)*cosf(2 * u)*cosf(v)*(-sinf(v))) - (cosf(u)*cosf(2 * v) * 2); ;
		float dg_y = dg_x;

		float f_z = (3 * pow(cosf(v), 2)); // Számláló
		float g_z = g_x;                 // Nevező
		float df_z = 6 * cosf(v)*(-sinf(v));
		float dg_z = dg_x;

		return vec3(
			fractionDrivated(f_x, df_x, g_x, dg_x), 
			fractionDrivated(f_y, df_y, g_y, dg_y), 
			fractionDrivated(f_z, df_z, g_z, dg_z)
		);
	}

	float fractionDrivated(float f, float df, float g, float dg) {
		return ((df*g) - (f*dg)) / (pow(g, 2));
	}

public:
	BoySurface() {
		Create();
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = (u - 0.5f) * M_PI;     // u \in [-PI/2, PI/2]
		float V = v * M_PI;              // v \in [0, PI]
		Clifford cx = (Cos(T(V))*Cos(T(V))*Cos(T(U)*2)*sqrtf(2.0) + Cos(T(U))*Sin(T(V)*2)) / (Sin(T(U)*3)*Sin(T(V)*2)*(-1)*sqrtf(2.0)+2);
		Clifford cy = (Cos(T(V))*Cos(T(V))*Cos(T(U) * 2)*sqrtf(2.0) - Cos(T(U))*Sin(T(V) * 2)) / (Sin(T(U) * 3)*Sin(T(V) * 2)*(-1)*sqrtf(2.0) + 2);
		Clifford cz = (Cos(T(V)*3))/(Sin(T(U)*3)*Sin(T(V)*2)*(-1)*sqrtf(2.0) + 2);
		vd.position = vec3(cx.f, cy.f, cz.f);
		vec3 drdU(cx.d, cy.d, cz.d);
		vd.normal = cross(drdU, drdV(U, V));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};


``` 
