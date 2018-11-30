// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Név: Gyõri Kristóf
// Neptun : HV0R9S
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 30;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
	float l;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 90 * (float)M_PI / 180.0f;
		fp = 1; bp = 50;
		
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
												   u.y, v.y, w.y, 0,
												   u.z, v.z, w.z, 0,
												   0,   0,   0,   1);
	}

	mat4 Vinv() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		
		return  mat4(u.x, u.y, u.z, 0,
					 v.x, v.y, v.z, 0,
					 w.x, w.y, w.z, 0,
					 0, 0, 0, 1)*TranslateMatrix(wEye * (1));
	}

	mat4 P() { // projection matrix
		l = tanf(fov / 2)*length(wLookat - wEye);
		/*return mat4(
			1/l, 0, 0, 0,
			0, 1/l, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1);*/
		return mat4(1 / (tan(fov / 2)*asp), 0, 0, 0,
						0, 1 / tan(fov / 2), 0, 0,
						0, 0, -(fp + bp) / (bp - fp), -1,
						0, 0, -2 * fp*bp / (bp - fp), 0);
	}
	mat4 Pinv() {
		l = tanf(fov / 2)*length(wLookat - wEye);
		return ScaleMatrix(vec2(l / 1, l / 1));
	}
	void Animate(float t) { }
};

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.kd", name);
		kd.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ks", name);
		ks.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.ka", name);
		ka.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.shininess", name);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform shininess cannot be set\n");
	}
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;

	void Animate(float t) {	}

	void SetUniform(unsigned shaderProg, char * name) {
		char buffer[256];
		sprintf(buffer, "%s.La", name);
		La.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.Le", name);
		Le.SetUniform(shaderProg, buffer);

		sprintf(buffer, "%s.wLightPos", name);
		wLightPos.SetUniform(shaderProg, buffer);
	}
};

//---------------------------
struct CheckerBoardTexture : public Texture {
	
	//---------------------------
	CheckerBoardTexture(const vec3 color1, const vec3 color2, const int width = 0, const int height = 0) : Texture() {
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		std::vector<vec3> image(width * height);
		
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? color1 : color2;
		}
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, &image[0]); //Texture->OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//--------------------------
public:
	virtual void Bind(RenderState state) = 0;
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { Create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		glUseProgram(getId()); 		// make this program run
		state.MVP.SetUniform(getId(), "MVP");
		state.M.SetUniform(getId(), "M");
		state.Minv.SetUniform(getId(), "Minv");
		state.wEye.SetUniform(getId(), "wEye");
		state.material->SetUniform(getId(), "material");

		int location = glGetUniformLocation(getId(), "nLights");
		if (location >= 0) glUniform1i(location, state.lights.size()); else printf("uniform nLight cannot be set\n");
		for (int i = 0; i < state.lights.size(); i++) {
			char buffer[256];
			sprintf(buffer, "lights[%d]", i);
			state.lights[i].SetUniform(getId(), buffer);
		}
		state.texture->SetUniform(getId(), "diffuseTexture");
	}
};

//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
//---------------------------
protected:
	unsigned int vao;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	virtual void Draw() = 0;

	
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() {
		nVtxPerStrip = nStrips = 0;
	}
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = tessellationLevel, int M = tessellationLevel) {
		unsigned int vbo;
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
									   // attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		for (int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
struct Clifford {
	//---------------------------
	float f, d;
	Clifford(float f0 = 0, float d0 = 0) { f = f0, d = d0; }
	Clifford operator+(Clifford r) { return Clifford(f + r.f, d + r.d); }
	Clifford operator-(Clifford r) { return Clifford(f - r.f, d - r.d); }
	Clifford operator*(Clifford r) { return Clifford(f * r.f, f * r.d + d * r.f); }
	Clifford operator/(Clifford r) {
		float l = r.f * r.f;
		return (*this) * Clifford(r.f / l, -r.d / l);
	}
};

Clifford T(float t) { return Clifford(t, 1); }
Clifford Sin(Clifford g) { return Clifford(sin(g.f), cos(g.f) * g.d); }
Clifford Cos(Clifford g) { return Clifford(cos(g.f), -sin(g.f) * g.d); }
Clifford Tan(Clifford g) { return Sin(g) / Cos(g); }
Clifford Log(Clifford g) { return Clifford(logf(g.f), 1 / g.f * g.d); }
Clifford Exp(Clifford g) { return Clifford(expf(g.f), expf(g.f) * g.d); }
Clifford Pow(Clifford g, float n) { return Clifford(powf(g.f, n), n * powf(g.f, n - 1) * g.d); }

//---------------------------
class BoySurface : public ParamSurface {
//---------------------------
private:
	float R, w;

	vec3 drdV(float u, float v) {
		float f_x = (sqrtf(2.0)*pow(cosf(v), 2)*cosf(2 * u) + cosf(u)*sinf(2 * v)); // Számláló
		float g_x = (2 - sqrtf(2.0)*sinf(3 * u)*sinf(2 * v));                         // Nevezõ
		float df_x = (2 * sqrtf(2.0)*cosf(2 * u)*cosf(v)*(-sinf(v))) + (cosf(u)*cosf(2 * v) * 2);
		float dg_x = -sqrtf(2.0)*sinf(3 * u)*cosf(2*v) * 2;

		float f_y = (sqrtf(2.0)*pow(cosf(v), 2)*cosf(2 * u) + cosf(u)*sinf(2 * v)); // Számláló
		float g_y = g_x;                                                        // Nevezõ
		float df_y = (2 * sqrtf(2.0)*cosf(2 * u)*cosf(v)*(-sinf(v))) - (cosf(u)*cosf(2 * v) * 2); ;
		float dg_y = dg_x;

		float f_z = (3 * pow(cosf(v), 2)); // Számláló
		float g_z = g_x;                 // Nevezõ
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

class PlaneXY : public ParamSurface {
public:
	float size;
	PlaneXY() {
		size = 100;
		Create(20, 20);
	}
	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(0, 0, 1);
		vd.position = vec3((u - 0.5) * 2, (v - 0.5) * 2) * size;
		vd.texcoord = vec2(u, v);
		return vd;
	}

};

vec4 qmul(vec4 q1, vec4 q2) {	// kvaternió szorzás
	vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
	vec3 temp = d2 * q1.w + d1 * q2.w + cross(d1, d2);
	return vec4(temp.x, temp.y, temp.z, q1.w * q2.w - dot(d1, d2));
}
vec3 Rotate(vec3 u, vec4 q) {
	vec4 qinv(-q.x, -q.y, -q.z, q.w);
	vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
	return vec3(qr.x, qr.y, qr.z);
}


//---------------------------
struct Object {
	//---------------------------
	Shader * shader;
	Material * material;
	Texture * texture;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	void Draw(RenderState state) {
		state.M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		state.Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
		state.MVP = state.M * state.V *state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8 * tend; }
};

class BoyObject : public Object {
public:
	BoyObject(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) : Object(_shader, _material, _texture, _geometry) { }
	void Animate(float tstart, float tend) {
		rotationAngle = tend*0.8;
		vec4 q = vec4(
			cosf(rotationAngle / 3.0),
			(sinf(rotationAngle / 3.0)*sinf(rotationAngle / 3.0)) / 2.0,
			(sinf(rotationAngle / 3.0)*sinf(rotationAngle)) / 2.0,
			(sinf(rotationAngle / 3.0)*sqrt(3.0 / 4.0)));

		rotationAxis = vec3(q.y, q.z, q.w);
	    //translation = Rotate(vec3(0, 0, 3), q);
	}
};

class DiniObject : public Object {

public:
	DiniObject(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) : Object(_shader, _material, _texture, _geometry)
    {
		
	}
	void Animate(float tstart, float tend) { 
		//rotationAngle = 0.8 * tend; 
	}
};
class PlaneXYObject : public Object {

public:
	PlaneXYObject(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) : Object(_shader, _material, _texture, _geometry)
	{

	}
	void Animate(float tstart, float tend) {
		//rotationAngle = 0.8 * tend; 
	}
};

//---------------------------
class Scene {
	//---------------------------
	std::vector<Object *> objects;
public:
	Camera camera; // 3D camera
	std::vector<Light> lights;

	void Build() {
		// Shaders
		Shader * phongShader = new PhongShader();
		
		// Materials
		Material * material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material * material1 = new Material;
		material1->kd = vec3(0.8, 0.6, 0.4);
		material1->ks = vec3(2, 2,2);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 100;

		Material * diffuse = new Material;
		//vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		diffuse->kd = vec3(0, 0.2, 0);
		diffuse->ks = vec3(0, 0, 0);
		diffuse->ka = vec3(0.8f, 0.8f, 0.8f);
		diffuse->shininess = 30;
		
		// Textures
		Texture * texture15x20 = new CheckerBoardTexture(vec3(1, 1, 1), vec3(0, 0, 0), 15, 20);
		Texture * textureFloor = new CheckerBoardTexture(vec3(0, 0.4, 0), vec3(0, 0.4, 0), 10, 10);
		Texture * textureTree = new CheckerBoardTexture(vec3(0, 1, 1), vec3(1, 1, 0), 10, 10);

		// Geometries
		Geometry * boy = new BoySurface();
		Geometry * dini = new DiniSurface();
		Geometry * plane = new PlaneXY();

		// Create objects by setting up their vertex data on the GPU
		Object * boyObject = new BoyObject(phongShader, material1, texture15x20, boy);
		boyObject->translation = vec3(0, 0, 2);
		boyObject->rotationAxis = vec3(0, 1, 1);
		boyObject->scale = vec3(0.7f, 0.7f, 0.7f);
		objects.push_back(boyObject);

		Object * diniObject1 = new DiniObject(phongShader, material0, textureTree, dini);
		diniObject1->translation = vec3(1,2);
		diniObject1->rotationAxis = vec3(0, 1, 1);
		diniObject1->scale = vec3(0.5f, 0.5f, 0.5f);
		objects.push_back(diniObject1);

		Object * diniObject2 = new DiniObject(phongShader, material0, textureTree, dini);
		diniObject2->translation = vec3(3, -2);
		diniObject2->rotationAxis = vec3(0, 1, 1);
		diniObject2->scale = vec3(0.5f, 0.5f, 0.5f);
		objects.push_back(diniObject2);

		Object * diniObject3 = new DiniObject(phongShader, material0, textureTree, dini);
		diniObject3->translation = vec3(0, -1, 0);
		diniObject3->rotationAxis = vec3(0, 1, 1);
		diniObject3->scale = vec3(0.5f, 0.5f, 0.5f);
		objects.push_back(diniObject3);
		
		Object * planeObject = new PlaneXYObject(phongShader, diffuse, textureFloor, plane);
		planeObject->translation = vec3(0, 0, -2);
		planeObject->rotationAxis = vec3(0, 1, 1);
		planeObject->scale = vec3(0.5f, 0.5f, 0.5f);
		objects.push_back(planeObject);

		
		// Camera
		camera.wEye = vec3(6, 0, 1);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 0 , 1);

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4(5, 5, 4, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0.1, 0.1, 1);
		lights[0].Le = vec3(1, 1, 1);

		lights[1].wLightPos = vec4(5, 10, 20, 0);	// ideal point -> directional light source
		lights[1].La = vec3(0.2, 0.2, 0.2);
		lights[1].Le = vec3(1,1, 1);

		lights[2].wLightPos = vec4(-5, 5, 5, 0);	// ideal point -> directional light source
		lights[2].La = vec3(0.1, 0.1, 0.1);
		lights[2].Le = vec3(1, 1, 1);

	}
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		camera.Animate(tend);
		for (int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
		for (Object * obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
	// Create objects by setting up their vertex data on the GPU

	
	
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { 
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   
		printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  
		//catmull.AddControlPoint(vec3(cX, cX, cY));
		break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}

	
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}