#define bpm 138.
#define pi acos(-1.)
#define pi2 pi*2.

float rand(vec2 p)
{
    return fract(sin(dot(p,vec2(12.9898,78.233)))*43478.5453);
}
float rand(float p)
{
    return fract(sin(p*78.233)*43478.5453);
}
float noise(float p)
{
    float i=floor(p);
    float f=fract(p);
    f=f*f*(3.-2.*f);
    return mix(rand(i),rand(i+1.),f);   
} 

float fbm(float p)
{
    float amp=.5,val;
    for(int i=0;i<4;i++)
    {
      val+=noise(p)*amp;
      p*=2.;
      amp*=.5;
    }
    return val;
}

float fm(float t,float f,float i,float r){
    return sin(pi2*f*t+i*sin(pi2*f*t*r));
}
float sfm(float t,float f,float i1,float r1,float i2,float r2){
    return sin(pi2*f*t+i1*fm(t,f,i2,r2)*r1);
}
float ssaw(float t,float f){
    return rand(vec2(t))*0.02+sfm(t,f,0.8,1.8,.8,7.0)+sfm(t,f*1.008,0.8,1.8,0.8,6.0)+0.1*sfm(t,f,0.1,0.0,0.1,3.0);
}
float calf(float i){
    return pow(2.0,i/12.0);
}

float sigcomp1d(float w,float s){
  return (1.0/(1.0+exp(-s*w)) -0.5)*2.0;
}

float dist(float s, float d)
{
    return clamp(s * d, -1.0, 1.0);
}

float metalic(float t,float time)
{
    float o = 0.;
    //o += mix(sin(time*pi2*440.*calf(4.)),fm(time,440.*calf(4.),1.,1.5),1.5)*.5;
    float[] amps = float[](1.,1.,.25,1.);
    float amp = amps[int(mod(t*.125*.25,4.))];
    o+=fm(time,440.*calf(1.+12.+12.),.35,4.5+floor(sin(t*pi2*2.)))*((noise(time*pi2*220.*calf(2.)*.75))*1.-.25);
    
    o = o * exp(-7.*fract((t*.5+.5)))*.7*amp;// * smoothstep(1.,.75,fract(t*.25));
    return o;
}   
float base(float t,float time)
{
    float o = 0.;
    float note = 4.5;
	o += fm(time,40.*calf(note),2.,1.);
    o += sin(time*40.*calf(note+12.)*pi2)*.5;
    //o += fm(time,40.*calf(5.+note),2.,1.);
    //o += fm(time,40.*calf(31.+note),2.,1.5);
    o *= .4;
    o = dist(o,2.)*.5;
    o = o * exp(-.75*fract(t*2.));
	return o;
}

float saw(float t,float time)
{
    float o = 0.;
    float[] notes = float[](-3.,-3.,-5.,-3.);
    float note = notes[int(mod(t*4.,4.))]+2.;
    float[] amps = float[](1.,1.,.5,1.);
    float amp = amps[int(mod(t*.125*.25,4.))];
    o += ssaw(time,220.*calf(0.+note))*.25;
    o += ssaw(time,220.*calf(-5.+note))*.25*amp;
    o += fm(time,220.*calf(-12.+note),1.,1.)*.5;
    o = o * exp(-.25*fract(t*2.))*.25;
	return o;
}

float hihat(float t,float time)
{
    float o = 0.;
    
    float[] notes = float[](-3.,-3.,-5.,-3.);
    float note = notes[int(mod(t*2.,4.))];

    o += sin(1760.*pi2*time*calf(24.+note))*.25;
    o += noise(time*1e4)+fbm(time*1e4)*.5+rand(time)*.5;
    o = o* exp(-11.*fract(t+.5))*.5; 
    //o = dist(o,2.)*1.;
	return o;
}

//kick
//https://www.shadertoy.com/view/ldfSW2
float kick(float t, float time)
{
    
    t = fract(t*1.)*.5;
    float aa = 5.;
    t = sqrt(t * aa) / aa; 	
    float amp = exp(max(t - 0.15, 0.0) * -10.0);
    
       
    float o = sin(t * 100.0 * pi2) * amp;
    o = dist(o,1.);
    return o;
}

float kickEcho(float t,float time)
{
    float amp = .5;
    float o = 0.;
    float[] kickamps = float[](0.,1.,0.,1.);
    float kickamp = kickamps[int(mod(t*.125*.25,4.))];
    for(int i=0;i<4;i++)
    {
      o+= kick(t+float(i)*.25,time)*amp;
      amp*=.5;
    }
    return o*kickamp;
}

vec2 mainSound(float time){
    float o = 0.;
    float t = time*bpm/60.;
    
    o += metalic(t,time);
    o += base(t,time);
    o += kickEcho(t,time);
    //o += hihat(t,time);
    o += saw(t,time);
    
    o = o*smoothstep(0.,4.,time);
    o = o*smoothstep(64.,56.,time);
    o*=1.0;
    return vec2(o*.5);
}