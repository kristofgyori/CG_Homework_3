# Dini's surface
``` cpp
class DiniSurface : public ParamSurface {
private:
	const float a = 1;
	const float b = 0.15;

public:
	DiniSurface() {
		Create();
	}

	vec3 r(float u, float v) {
		float x = a*cosf(u)*sinf(v);
		float y = a*sinf(u)*sinf(v);
		float z = a*(cosf(v) + logf(tanf(v / 2.0))) + b*u;
		return vec3(x, y, z);
	}

	vec3 dr(float u, float v) {
		float rv_x = a*cosf(u)*cosf(v);
		float rv_y = a*sinf(u)*cosf(v);
		float rv_z = a*sinf(v) + (1.0 / (2 * sinf(v / 2.0)*cosf(v / 2.0)));

		float ru_x = a*sinf(v)*(-sinf(u));
		float ru_y = a*sinf(v)*cosf(u);
		float ru_z = b;

		vec3 drdv(rv_x, rv_y, rv_z);
		vec3 drdu(ru_x, ru_y, ru_z);

		return cross(drdu, drdv);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		float U = u* 4*M_PI;     // u \in [0, 4PI]
		float V = (v+0.01 ) *1;              // v \in [0.01, 1]
		vd.position = r(U, V);
		vd.normal = dr(U, V);
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

```
