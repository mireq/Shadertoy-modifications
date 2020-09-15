#define MAT_SPHERE 0.0
#define MAT_FLOOR 1.0
#define MAT_BOX 2.0
#define bpm 138.
#define pi acos(-1.)
#define pi2 pi*2.
#define rep(p,r) mod(p,r)-r*.5;
#define repid(p,r) (floor((p) / r))
#define saturate(x) clamp(x,0.,1.)

#define beat iTime*bpm/60.
#define speed 16.

float rand(vec2 p)
{
	return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);   
}

float easeIn(float t) {
	return t * t;
}

float easeOut(float t) {
	return -1.0 * t * (t - 2.0);
}

float rand(float p)
{
	return fract(sin(p*43758.5453));  
}

float noise(vec2 p)
{
	vec2 i=floor(p),f=fract(p);
	f=f*f*(3.-2.*f);
	return mix(
		mix(rand(i),rand(i+vec2(1.,0.)),f.x),
		mix(rand(i+vec2(0.,1.)),rand(i+vec2(1.)),f.x),
		f.y);
}

float fbm(vec2 p)
{
	float amp=.5,val=0.0;
	for(int i=0;i<4;i++)
	{
		val +=noise(p)*amp;
		amp *= .5;
		p*=2.;
	}
	return val;
}

mat2 rot(float r)
{
	float s=sin(r),c=cos(r);
	return mat2(c,s,-s,c);
}

vec2 pmod(vec2 p, float r) {
	float a = pi/r - atan(p.x, p.y);
	float n = pi2/r;
	a = floor(a/n) * n;
	return p * rot(a);
}

float luminace(vec3 col)
{
	return dot(vec3(0.298912,0.58611,0.114478),col);   
}

vec3 saturation(vec3 col,float scale)
{
	return mix(vec3(luminace(col)),col,scale);   
}

vec3 contrast(vec3 col,float scale)
{
	return (col-.5)*scale+.5;
}

vec3 colorCorrect(vec3 col){
	col = saturation(col,1.4);
	col = pow(col,vec3(2.5));
	//col = pow(col,vec3(1.5));
	col = col*vec3(.6,.9,1.0);
	return col;
}

vec3 acesFilm(const vec3 x) {
	const float a = 2.51;
	const float b = 0.03;
	const float c = 2.43;
	const float d = 0.59;
	const float e = 0.14;
	return clamp((x * (a * x + b)) / (x * (c * x + d ) + e), 0.0, 1.0);
}

float vignette(vec2 p,float s){
	p *= 1.-p.yx;
	float vig = p.x*p.y*30.;
	vig = clamp(pow(vig,s),.0,1.);
	return vig;
}

float sphere(vec3 p, float s) {
	return length(p) - s;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

vec2 opU(vec2 d1, vec2 d2)
{
	return (d1.x<d2.x) ? d1 : d2;
}

vec2 opS(vec2 d1,vec2 d2)
{
	return (d1.x>d2.x) ? d1 : d2;
}

vec2 map(vec3 p)
{
	
	float t = iTime;
	float start = step(0.,beat);
	float end=saturate(beat-120.);
	vec3 q = p;
	
	vec3 sp = vec3(0.,0.,-t*speed);
	
	q+=sp;
	
	float l1 = easeOut(1.-clamp(7.*.2-length(q.xz*.05)-t*.2,0.,1.));
	float l2 = easeOut(clamp(64.*.8-length(q.xz*.05)-t*.8,0.,1.));
	float l = 1.;
	l = mix(l,l1,start);
	l = mix(l,l2,end);
	float d1 = sphere(q,.5*l);
	
	q=p;
	
	q+=sp;
	for(int i=0;i<2;i++)
	{
		q = abs(q)-.25;
		q.xz *= rot(1.);
		q.xy *= rot(1.+iTime);
		
	}
	float d2 = sdBox(q,vec3(.125,.5,.124)*l);
	d1 = mix(d1,d2,pow(sin(fract(beat)*pi)*.5+.5,8.));
	vec2 s = vec2(d1,MAT_SPHERE);
	
	q=p;
	q= -abs(q);
	vec2 fl = vec2(q.y+10.5*l+.05,MAT_FLOOR);
	
	//vec2 fl2 = vec2 (q.x+10.5*l+.05,MAT_FLOOR);
	q=p;
	q.xz = rep(q.xz,2.);
	
	//fl = opS(fl,fl2);
	vec2 tri = vec2(1.);
	
	//q.xy *= rot(.2);
	
	q = p;
	q.xy = pmod(p.xy,3.);
	q.z = rep(p.z,10.);
	vec2 box = vec2(sdBox(q-vec3(0.,5.,0.),vec3(.5)),MAT_BOX);
	
	q = p;
	
	q.y = abs(q.y)-2.1;
	q.xz = rep(q.xz,14.);
	for(int i=0;i<4;i++)
	{
		q = abs(q)-2.4;
		q.xy *= rot(2.025+(1.-l));
		
	}
	
	vec2 w = vec2(sdBox(q,vec3(2.,1.2,.2)),MAT_BOX);
	q=p;
	vec2 d = opU(s,fl);
	//d = opU(d,box);
	d = opU(d,w);
	return d;
}

vec4 volMap(vec3 p)
{
	vec3 q = p;
	q.xy *= rot(pi*.25);
	float zid = repid(q.z,10.);
	q.z = rep(q.z,10.);
	
	q = abs(q);
	q.xy =abs(q.xy)-5.;
	float d = length(q.xz)-.05;
	d = min(d,length(q.yz)-.05);
	float r0 = rand(zid);
	vec3 col = vec3(.2,.8,.8)*(.1+0.9*exp(-13.*fract(beat*4.)))*.01;
	return vec4(col,d);
}

vec4 volMap2(vec3 p)
{
	vec3 q=p;
	q.xy = pmod(q.xy,3.);
	//float d = sdCappedCylinder(q-vec3(0.,5.,0.),.1,2.1);
	q.y -=5.;
	
	float d = length(q.yx)-.1;
	
	
	vec3 col = vec3(.9,.2,.2)*(.00+1.*exp(-13.*fract(beat*.5)))*.2;
	return vec4(col,d);
}

vec4 volMap3(vec3 p)
{
	vec3 q=p;
	q.xy *= rot(pi);
	q.xy = pmod(q.xy,3.);
	//float d = sdCappedCylinder(q-vec3(0.,5.,0.),.1,2.1);
	q.y -=5.;
	float d = length(q.yx)-.1;
   
	
	vec3 col = vec3(.2,.2,.9)*(.00+1.*exp(-13.*fract(beat*.5+.5)))*.2;
	return vec4(col,d);
}

vec4 volMap4(vec3 p)
{
	vec3 q=p;
	
	//q.xz = rep(q.xz,14.);
	
	//float d = sdCappedCylinder(q-vec3(0.,5.,0.),.1,2.1);
	//q.y =abs(q.y)-1.7;
	vec3 id = repid(q,7.);
	q.z = rep(q.z,7.);
	
	q.y = abs(q.y)-8.;
	float d = length(q.zy)-.025;
   
	
	vec3 col = vec3(.6,.4,.7)*.025*(0.01+1.*exp(-13.*fract(beat+.5)));
	return vec4(col,d);
}

vec4 volMap5(vec3 p)
{
	vec3 q=p;
	vec3 sp = vec3(0.,0.,-iTime*speed);
	q+=sp;

	
	float d = length(q)-.125;
   
	
	vec3 col = vec3(.98,.2,.2)*.025;//*(0.01+1.*exp(-1.*fract(beat)));
	return vec4(col,d);
}

vec3 normal(vec3 p)
{
  float e = 0.001;
  vec2 k = vec2(1.,-1.);
  return normalize(
	  k.xyy * map(p+k.xyy*e).x+
	  k.yxy * map(p+k.yxy*e).x+
	  k.yyx * map(p+k.yyx*e).x+
	  k.xxx * map(p+k.xxx*e).x
	);
}

float shadow(in vec3 p, in vec3 l)
{
	float t = 0.01;
	float t_max = 20.0;
	
	float res = 1.0;
	for (int i = 0; i < 64; ++i)
	{
		if (t > t_max) break;
		
		float d = map(p + t*l).x;
		if (d < 0.001)
		{
			return 0.0;
		}
		t += d;
		res = min(res, 10.0 * d / t);
	}
	
	return res;
}


vec3 glow(vec3 ro,vec3 rd,float depth)
{
  float t = 0.01;
  vec4 d = vec4(0.0);
  float sd = depth/99.;
  vec3 ac = vec3(0.0);
  float[] parts = float[](0.,1.,0.,1.);
  float part = parts[int(mod(beat*.125*.25,4.))];
	
	float[] parts2 = float[](1.,1.,0.,1.);
  float part2 = parts2[int(mod(beat*.125*.25,4.))];
  for(int i = 0;i<60;i++)
  {
	  if(t>depth) break;
	  vec3 pos = ro+rd*t;
	  d = volMap(pos);
	  d.w = max(0.01,d.w);
	  ac += (d.rgb*sd)/(pow(d.w,2.))*part2 * smoothstep(4.,12.,iTime);
	  
	  
	  d = volMap2(pos);
	  d.w = max(0.01,d.w);
	  ac += (d.rgb*sd)/(pow(d.w,2.))*part;
	  
	  d = volMap3(pos);
	  d.w = max(0.01,d.w);
	  ac += (d.rgb*sd)/(pow(d.w,2.))*part;
	  
	  d = volMap4(pos);
	  d.w = max(0.01,d.w);
	  ac += (d.rgb*sd)/(pow(d.w,2.));
	  
	  d = volMap5(pos);
	  d.w = max(0.01,d.w);
	  ac += (d.rgb*sd)/(pow(d.w,2.));
	  
	  t += sd;
  }
  return ac;
}


float ndfGGX(float NdotH, float roughness)
{
	float alpha   = roughness * roughness;
	float alphaSq = alpha * alpha;

	float denom = (NdotH * NdotH) * (alphaSq - 1.0) + 1.0;
	return alphaSq / (pi * denom * denom);
}

float gaSchlickG1(float cosTheta, float k)
{
	return cosTheta / (cosTheta * (1.0 - k) + k);
}

float gaSchlickGGX(float NdotL, float NdotV, float roughness)
{
	float r = roughness + 1.0;
	float k = (r * r) / 8.0;
	return gaSchlickG1(NdotL, k) * gaSchlickG1(NdotV, k);
}

vec3 fresnelSchlick(vec3 F0, float cosTheta)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 fresnelSchlickWithRoughness(vec3 F0, float cosTheta, float roughness) {
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

//Kaneta SkyColor 
//https://www.shadertoy.com/view/3lXfzS
vec3 skyColor(vec3 rd, float roughness)
{
	vec3 baseColor = mix(vec3(0.3,0.5,0.8)*0.8, vec3(0.3,0.5,0.8) * 0.5, rd.y);
	baseColor = mix( baseColor, vec3(0.2,0.5,0.85)*0.5, 1.0 - pow(1.0-max(-rd.y,0.0), 1.5));
	vec3 skyColor = baseColor;
	skyColor = mix( skyColor, vec3(0.9,1.1,1.2) * 1.5, pow( 1.0-max(rd.y,0.0), 8.0 ) );
	skyColor = mix( skyColor, vec3(0.2,0.5,0.85)*0.2, 1.0 - pow(1.0-max(-rd.y,0.0), 6.0));
	
	return mix(skyColor, baseColor, pow(roughness, 0.1)) * 10.0;
}

float so(float NoV, float ao, float roughness) {
	return clamp(pow(NoV + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

vec3 ambientLighting(vec3 pos, vec3 albedo, float metalness, float roughness, vec3 N, vec3 V, float aoRange)
{
	vec3 diffuseIrradiance = skyColor(N, 1.0);
	
	vec3 diffuseAmbient = diffuseIrradiance * albedo * (1.0 - metalness);

	vec3 R = reflect(-V, N);
	vec3 F0 = mix(vec3(0.04), albedo, metalness);
	vec3 F  = fresnelSchlickWithRoughness(F0, max(0.0, dot(N, V)), roughness);
	vec3 specularIrradiance = skyColor(R, roughness);
	vec3 specularAmbient = specularIrradiance * F;

	float ambientOcclusion = max( 0.0, 1.0 - map( pos + N*aoRange ).x/aoRange );
	ambientOcclusion = min(exp2( -.8 * pow(ambientOcclusion, 2.0) ), 1.0) * min(1.0, 1.0+0.5*N.y);
	diffuseAmbient *= ambientOcclusion;
	specularAmbient *= so(max(0.0, dot(N, V)), ambientOcclusion, roughness);

	return vec3(diffuseAmbient + specularAmbient);
}

vec3 directLighting(vec3 pos, vec3 albedo, float metalness, float roughness, vec3 N, vec3 V, vec3 L, vec3 lightColor)
{
	vec3 H = normalize(L + V);
	float NdotV = max(0.0, dot(N, V));
	float NdotL = max(0.0, dot(N, L));
	float NdotH = max(0.0, dot(N, H));
	float HdotL = max(0.0, dot(H, L));
		
	vec3 F0 = mix(vec3(0.04), albedo, metalness);

	vec3 F  = fresnelSchlick(F0, HdotL);
	float D = ndfGGX(NdotH, roughness);
	float G = gaSchlickGGX(NdotL, NdotV, roughness);
	vec3 specularBRDF = (F * D * G) / max(0.0001, 4.0 * NdotL * NdotV);

	vec3 kd = mix(vec3(1.0) - F, vec3(0.0), metalness);
	vec3 diffuseBRDF = kd * albedo / pi;
	
	//float shadow = shadow(pos + N * 0.01, L);
	vec3 irradiance = lightColor * NdotL;// * shadow;

	return (diffuseBRDF + specularBRDF) * irradiance;
}

vec3 lighting(vec3 ro,vec3 p,vec3 ray ,float depth,vec2 mat)
{
	vec3 col = vec3(0.0, 0.0, 0.0);
	
	vec3 n = normal(p);
	float start = saturate(iTime-2.5);
	vec3 ld = normalize(vec3(1.,1.,.2));
	float NdotL = max(0.,dot(n,ld));
	float metalness = .0;
	float roughness = .0;
	vec3 albedo = vec3(.5,.5,.5);
	float checker = mod(floor(p.x)+floor(p.z),2.);
	if(depth>200.)
	{
		return vec3(.1);   
	}
	else if(mat.y == MAT_SPHERE){
		albedo = vec3(.25);
		roughness = .1;
		metalness = 1.;
	} else if(mat.y == MAT_FLOOR){
		albedo = vec3(.025);
		metalness = .5;
		roughness = .8;
	} else if(mat.y == MAT_BOX){
		albedo = vec3(.5,.5,.5);
		metalness = 1.;
		roughness = 0.1;
	} else {
		col = vec3(.1);
	}
	float aoRange = depth/30.;
	
	col += directLighting(p, albedo, metalness, roughness, n, -ray, normalize(ld), vec3(.98, 0.18, 0.18) * 10.*(sin(beat*pi*.25)*.5+.5));
	
	col += ambientLighting(p, albedo, metalness, roughness, n, -ray, depth / 30.0)*.05;
	
	col += glow(ro,ray,depth);
	float fog = exp(-.0006*depth);
	
	vec3 fog2 = .02*vec3(1.,1.,1.4)*depth;
	fog2 *= start*.5;
	col = mix(vec3(.0),col+fog2,fog);
	return col;
}

void trace(vec3 ro,out vec3 p, vec3 ray,out float t,out vec2 mat)
{
	t=0.01;
	vec3 pos = vec3(0.0, 0.0, 0.0);
	for (int i = 0; i < 99; i++) {
		pos = ro + ray * t;
		mat = map(pos);
		if (mat.x < 0.01) {
			break;
		}
		t += abs(mat.x);
	}
	p = pos;
}

vec3 ray(vec2 p,vec3 ro,vec3 ta,float fov)
{
	vec3 fo = normalize(ta - ro);
	vec3 si = normalize(cross(vec3(0.,1.,0.),fo));
	vec3 up = normalize(cross(fo,si));
	return normalize(fo*fov+si*p.x+up*p.y);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	
	vec2 uv = fragCoord.xy /iResolution.xy;
	vec2 p = (fragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);
	
	float time = iTime*speed;
	vec2 fbm = vec2(fbm(vec2(iTime*.1)),fbm(vec2(iTime*.1+100.)))*2.-1.;
	vec3 camPos1 = vec3(0.,0.,-5.+time);
	vec3 camPos2 = vec3(0.,-1.,2.+time);
	vec3 camPos3 = vec3(.6,2.,-3.+time);
	vec3 camPos4 = vec3(7.,-1.,-8.+time);
	vec3[] Poss = vec3[](camPos1,camPos2,camPos3,camPos4);
	
	vec3 ro = vec3(0.,-1.,2.+time);
	ro = mix(Poss[int(mod((beat-8.)*.125*.25,4.))],Poss[int(mod((beat-8.)*.125*.25+1.,4.))],smoothstep(0.,1.,saturate(mod(beat-8.,32.)-24.)));
	ro.xy += fbm*2.;
	vec3 ta = vec3(0.,0.,0.+time);
	
	vec3 ray = ray(p,ro,ta,1.2);

	vec3 pos = vec3(0.0, 0.0, 0.0),rpos = vec3(0.0, 0.0, 0.0);
	vec2 mat = vec2(0.0, 0.0);
	float t=0.01;
	trace(ro,pos,ray,t,mat);
	vec3 col = lighting(ro,pos, ray, t, mat);
	vec3 n = normal(pos);
	
	ray = reflect(ray,n);
	trace(pos,rpos,ray,t,mat);
	col += lighting(pos,rpos+n*.01, ray, t, mat)*.5;
	col = colorCorrect(col);
	col = acesFilm(col*.3);
	col = pow(col,vec3(1./2.2));
	col *= vignette(uv,.5);
	
	//col *= step(abs(p.y),.8);
	fragColor = vec4(col,1.0);
}
