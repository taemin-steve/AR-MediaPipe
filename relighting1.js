const cameraElement = document.getElementsByClassName('input_camera')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

import * as THREE from './node_modules/three/build/three.module.js';
import {OrbitControls} from './node_modules/three/examples/jsm/controls/OrbitControls.js'; //import map 사용해야 사용이 가능하다
import { Vector3 } from 'three';
import { Vector2 } from 'three';

function ProjScale(p_ms, cam_pos, src_d, dst_d) { //scale을 바꾸어 주는 부분
    let vec_cam2p = new THREE.Vector3().subVectors(p_ms, cam_pos);
    return new THREE.Vector3().addVectors(cam_pos, vec_cam2p.multiplyScalar(dst_d/src_d));
    
}

function background1_sphere(){
  let geomEnv = new THREE.SphereGeometry(50, 64, 64);
  // texture
  let loader = new THREE.TextureLoader();
  let textureEnv = loader.load('Museuminterior.jpg');
  textureEnv.mapping = THREE.EquirectangularReflectionMapping;
  textureEnv.encoding = THREE.sRGBEncoding;
  let material = new THREE.MeshLambertMaterial({
    map: textureEnv,
    side: THREE.BackSide
  });
  
  earth = new THREE.Mesh(geomEnv, material);
  console.log(earth);
  scene.add(earth);

}


function knn(features, labels, predictionPoint, k) {
    const { mean, variance } = tf.moments(features, 0);
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
    return (
      features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        // .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1))
        .sort((a, b) => a.arraySync()[0] - b.arraySync()[0])
        // .sort((a, b) => a.array().then((value) => {
        //   console.log(value[0]);}) - b.array().then((value) => {
        //     console.log(value[0]);}))
        .slice(0, k)
        // .reduce((acc, pair) => acc + pair.get(1), 0) / k
    );
}

const features = tf.tensor([

  // [35.255824,12.681438,12.162714,1.949913,4.178936,2.69438,5.003759,5.714465,4.993314,2.866081,7.437629,5.021466,6.839853,3.950859,7.966758],
  // [35.381951,6.5255,19.199896,3.555485,5.347654,2.759508,1.414179,4.648061,6.142657,1.858605,7.096097,7.531731,6.483093,6.014813,12.057738],
  // [41.645858,15.220567,8.432321,3.891638,2.995404,3.117573,2.547617,4.607339,4.529059,4.610331,8.58259,3.758644,12.137491,3.805865,7.507453],
  // [31.975783,6.425621,27.08857,1.623824,0.3246,3.288059,4.708754,6.496409,2.555214,3.167269,10.5684,3.97749,5.989966,12.268311,8.626645],
  // [52.354737,15.120519,8.159867,12.150277,3.072984,2.300643,6.89706,7.296752,5.010795,13.108232,14.79728,6.259755,33.110258,40.388509,18.43561],
  // [42.999924,9.240689,12.167061,6.017784,5.659045,3.776685,8.656432,8.790815,2.774698,9.226938,7.683244,2.427294,5.585492,5.957392,7.547471],
  // [43.477665,14.105631,13.982119,10.78948,3.81061,5.423428,6.094256,6.136796,5.340599,8.386305,7.252753,1.856217,5.491802,10.913479,5.328268],
  // [20.702271,10.081805,4.982003,23.877205,7.948928,1.551362,19.273723,11.337202,3.647189,20.742387,9.034537,1.285506,19.839281,4.202442,2.836296],
  // [17.150237,8.94054,9.556369,14.012305,6.129931,4.739453,8.193098,13.911509,5.34465,6.554458,10.177747,5.336161,3.375271,4.698845,3.42079],
  // [39.389713,1.117582,18.51818,2.084555,9.091669,5.572484,4.861925,9.557979,1.797316,8.672537,6.123965,4.182991,5.379867,3.624339,9.622453],
    
  [68.82650614378153,70.51215562787038,55.16470077198236,95.90697683849923,136.7870330132008,35.43515301414659,148.39176755249966,44.680978952184795,155.8522483764853,44.295461142799354,146.0922256257353,43.47131089761206,154.24847085346238,41.687760313151685,140.76262062159037],
  [55.454385995032794,108.68452768698063,49.31608055097289,170.19459145972687,137.84594497987405,101.13172400566357,132.942539635229,112.41277241727911,139.86703749929129,105.62520241754406,126.92743095958636,119.54065018692094,134.43702565892352,110.10229636356105,119.72843240081606],
  [73.03181990044222,115.64379578379689,69.19089905586824,97.56390801465378,128.4876183650806,130.45034506535643,128.75028084004734,128.9511764686799,123.4981559954106,143.90063909847177,132.88208831944328,144.46124719728752,133.56791728772356,150.72513896997245,131.69712017221238],
  [80.26834224872233,120.83269331489426,43.10710235821273,118.26071509176327,152.52182340197297,137.1911723170151,131.59899212954383,136.12366121519676,130.49292156696447,125.93933428504367,130.94744170720136,134.01843552444691,130.85350763937646,143.6318378893093,133.15470676062813],
  [65.43118569702834,129.69611064333017,44.36878819838782,123.80680981746588,147.46366345222657,132.64495038248174,133.42984540501146,132.93823679427598,131.93736524377294,119.18172421178855,129.5663371868708,126.62131485212736,130.4740267175245,141.21463124883513,131.73356874377737],
  [43.58908821618652,112.96305222805739,60.76780912790341,132.76139092857005,144.72921458882786,119.72696992436144,138.54926873102187,111.67131061369679,134.30973025824787,159.08026617086944,135.53844703345874,156.15536951824558,133.0203834769003,157.75853371378906,133.1625960163217],
  [49.501632015690674,105.0463307332828,89.03896069791885,94.81424562864187,116.335510577338,143.9982526052274,134.4146562685002,143.28354785089533,128.02482923854106,151.8510028983671,138.09878794968733,150.17510656761692,133.02915290746844,163.3413814442492,134.95328034413703],
  [59.61065972993083,112.45505961714326,33.06554385229372,101.16634926349579,153.7199906231545,161.18749049230226,113.52572452271126,137.44995758874623,119.28452385394777,138.35677168547517,116.5231713452177,125.6098525864884,121.91922632451825,121.29698507151174,119.29295967726677],
  [49.438713373845815,139.47780305369582,45.07142323606006,141.88891189776305,154.1614757126646,120.23675332119156,138.3626755588675,115.45290850293321,136.8831679043484,143.3874210993412,133.4628093448602,140.87050855147947,132.33789897174566,158.6441296361402,128.11542643292972],
  [57.616191418310194,138.3830727810162,40.31594479046158,140.52916008315935,154.05048529489093,125.43341983197116,139.73738960614415,119.92466911531814,137.85755147128168,162.062621624301,133.83572815439882,158.28691870379572,134.35623008674884,166.37601272032495,132.05428687294628],
  
  [26.028225228544837,5.541932528377451,115.85836612982781,33.27058710316778,94.56781990600469,2.3868680844411263,166.8370480889554,10.210321252789836,156.52866392213923,2.377376360752248,171.0040773536498,5.4873008500096105,163.2783363337559,3.058652392461129,169.47103139152202],
  [34.350888991431454,30.3548100429811,94.98652345690412,36.057266638842044,90.85070661303992,9.697991473495401,171.56827235953585,19.561578823003366,163.0251821258113,3.726137797416811,172.07134273450015,8.126183607619327,169.89383127275093,1.4881260076333613,172.54967979181993],
  [24.6795048782835,6.966356259808743,130.11331259763347,34.83390941323239,109.05291393395757,4.88109490297115,164.20888447444415,10.548197127260849,157.483253074327,2.31233348330595,169.06997685608638,3.3684746216642676,159.65436484893172,5.520014569134777,167.80556901531304],
  [16.609679796019343,20.105453070749615,117.41698967254386,36.629301823689545,93.49450486916132,3.2874423768900534,171.53215207183678,7.91701741785662,169.63442498024904,2.276213429031026,171.87624096826326,6.122745184576604,171.46636519265684,5.313287945564457,171.8604792745304],
  [24.55704462633771,19.976396290343313,121.84938614194827,62.8945011219007,16.39540581191073,110.0357973627676,122.91080572613264,79.93945454883196,87.0437700613654,123.1525401029506,127.36871600973866,114.80729660150017,103.9846736350001,128.14260267571365,110.49622114644214],
  [10.941333725733996,11.556785306663567,144.79948116787816,38.82313334965008,112.73887512445008,5.5902644533326535,167.2892533473481,10.589207150280053,153.84057921679263,5.526764718857099,167.9186007999277,19.315729958947333,142.86077332056576,10.159624533838649,159.41415341636116],
  [33.45368343687777,4.892326624130623,114.71493936040511,24.67908482679715,101.93497877159234,0.16688357480183796,166.654009755148,10.925964672268307,157.3577341025638,0.9358244683058831,170.51783483759843,6.728536911885854,164.02525709029237,4.2903786428722315,169.42164269587312],
  [36.53348057209563,18.05231595459098,112.07368964308701,9.660107823992353,107.70005748147582,8.970006190828299,167.8202662955326,17.7133620029825,159.3205030367062,6.260341462169743,171.56413526773022,9.917384345610381,166.79682376772297,6.382647879612327,170.48205147066008],
  [29.22303771527762,3.6901256194352254,120.2116816803865,24.653033312032125,109.22969791450181,4.7422900422313115,167.96317019516,16.195832129018047,155.88718964835564,2.1715992964101014,171.1872146164594,10.220117039544759,162.60223136105114,7.497097399214777,168.56737877852447],
  [29.43568885927946,6.0168442040194,112.93517082640419,25.035058421732803,109.87081938818336,13.771148792023096,169.81913823924924,24.551880653352022,161.00995671778426,10.677804787817184,173.03630475038798,15.814681712823813,169.9154054194688,11.474063800264673,173.42723906767046],

  [28.0965494791641,14.436161495288774,131.76912411266306,97.71513194954757,96.53128455472994,121.41749745056413,120.78037629518991,109.49027081154419,117.98615930532189,137.03936704429682,121.98839622495677,136.36928985845842,119.710182677649,147.03121499306266,121.37064462605747],
  [21.552597016461352,14.201952975543799,156.6476511639226,80.11569429131407,32.73904012012493,157.89314381386285,28.195438696222944,163.93332469975527,9.560375219512238,167.72936733144607,38.33538323318651,170.70744944265863,58.86954475409284,162.815802126131,78.42127873926452],
  [34.30469156487821,13.11153318076704,127.5036856364609,125.44381467794597,96.85103696247832,96.07898337558963,142.6689572418673,89.3637746057605,140.84367996018602,145.92810764785787,145.5591274605044,141.748338112614,142.80722989660865,156.4895626043805,140.56705451777557],
  [19.002447933802156,11.951950430005027,131.68330386109415,56.71443800081066,97.92592667106749,87.72102118761151,132.6322346642581,85.39989321132134,137.51804814178064,141.79524123695077,133.16974521966077,138.4144181903313,136.50575154610095,165.50765251075308,132.3557388168537],
  [12.747879103479018,5.472704747055402,138.57146279156692,104.57527954468688,115.93942420864212,48.40411626088832,149.6103183069122,55.687308984042595,150.4100980447071,111.80569851034028,155.3442968749642,108.71499978648126,151.0201566456159,123.89703325174426,155.46425184268966],
  [28.44101152489173,12.96582966584732,125.70096425505969,47.17093481825541,98.68539369693517,112.0447665640356,132.45599945825552,107.02117392904267,136.748046626066,159.61357145794176,132.1897647508915,153.83331414330084,137.7123999927397,170.7035258230984,134.55224967007075],
  [33.55746959791618,7.568639332169116,128.2482315215565,21.598923094737025,86.46682238877695,132.7105895408644,119.94748627913917,123.86727518119092,124.40754736631818,146.05113544498968,123.41037749226096,138.63265744103114,125.76132415334966,166.41040994268104,123.69276993078103],
  [27.51255888603711,11.155165271582016,136.91592675912128,69.69847945944497,79.99755427626911,83.28046418322967,110.12791333027056,82.07501305849789,109.81302434593775,143.75873730065442,109.17438871258501,141.52530531036882,106.52887187784067,168.51177680754873,105.00528831574033],
  [36.93585872389947,7.293156405744157,118.62160245151526,90.26553828170658,106.1460775158537,44.10501873803043,144.7872700424261,27.900483358410977,146.95772380977135,163.9110863048433,146.4000249851476,151.00520565682635,146.54012632750417,153.1793321738754,143.62825029313856],
  [42.06314434294726,11.64377481194526,132.31037288596295,58.8205639843048,75.20634907450363,135.05194690869888,115.78782649539154,130.22353852977793,118.84651730513085,151.11631176970758,113.6546233553085,152.88645823703243,113.2178051013385,164.67985081919315,112.45325759974371],

  [33.36073609207003,14.005946718186687,105.2183760476739,8.090850911522514,103.03053611857278,9.361620363203889,170.2263569701791,75.72206229648476,27.27649014541405,132.04079434493616,129.47353133748763,127.52659361804233,119.43989229677273,139.8563630854482,123.95977305444254],
  [39.27796469348973,99.1959814872592,70.47786112571157,74.00174403463527,63.703286750497,2.796289403243417,169.98795684563459,73.02505685849358,25.11033879144907,131.05388408737818,139.98311584479296,116.03794051846981,132.35726469672997,137.59287540956,136.02855235177066],
  [26.254791758099785,35.483674873257264,99.46438008874158,18.73957411105976,90.9818379489758,8.069984718845516,172.83905042383364,30.177617006728568,57.61923684398263,134.9409459966279,127.1958330883848,109.44720042741322,108.34366837045606,132.02413253968965,129.28961065229313],
  [24.707936340107892,17.78783450317176,107.290480970004,15.521833944312535,100.89467245957114,8.339977115899925,171.77459434918978,31.964824557992017,65.3939529254972,119.4600275027082,126.73401694934823,101.40061412999692,106.64200380644418,128.183635170679,124.53942332013995],
  [28.5652411836427,73.30361757185395,82.76225948783325,50.81060463381292,73.1241036777714,9.076366118690276,170.1084212729753,29.732239937318766,42.48161165739948,142.23835045033377,124.64923318335269,111.47759159794828,113.51911565833808,135.04044950332732,126.85438610126104],
  [29.596958295077687,13.013571586273942,110.66306467207399,15.311197661708695,118.19721602049322,4.42720868095268,175.01741424155549,26.442534066550632,60.82406640001619,124.82110711792168,122.97860412965655,106.11889776831711,104.05627847015562,128.97760231323608,123.81517165379032],
  [26.477340826587245,42.96540663339402,85.20975888244654,32.50099923839714,81.2282034629646,3.4592830148562403,170.29386402165287,69.2937396868471,33.07044581373195,116.625902693405,133.0552009907777,107.89052157318591,123.44154641613656,126.01060446139009,125.21606035429492],
  [42.79906647872732,31.113399029610967,158.1151725055031,17.01273805439617,153.45626539637,6.629407954530687,168.77704975212674,21.39739061566699,16.517861843654067,171.3449690019621,25.889686261953877,164.35112607652144,34.1990008570024,159.875367391014,70.75017748195427],
  [45.70166259719749,47.11203335927008,137.69209159719838,21.91307695871785,129.93524124356773,15.948393199025961,176.01322518245823,38.639058280201404,38.78076201041711,152.4153098505375,103.0966667678654,138.03061978926448,92.30803832282035,155.8922280176139,108.12537536092508],
  [25.475303914386366,13.56680152937759,108.63075975302912,16.045870748420473,107.02867352746934,10.78350071602639,172.10048674209634,30.7705800894018,63.57895913287959,122.3724706487572,125.50483419198476,102.81202690124103,108.41544628103385,130.8076716398686,126.17601985397549]
]);

const labels = tf.tensor([
  [0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
  [1],[1],[1],[1],[1],[1],[1],[1],[1],[1],
  [2],[2],[2],[2],[2],[2],[2],[2],[2],[2],
  [3],[3],[3],[3],[3],[3],[3],[3],[3],[3]
]);


const renderer = new THREE.WebGLRenderer();
const renderer_WorldView = new THREE.WebGLRenderer();


const renderer_w = 640;
const renderer_h = 480;
renderer.setSize( renderer_w, renderer_h ); 
renderer.setViewport(0,0,renderer_w, renderer_h); 

renderer_WorldView.setSize( renderer_w, renderer_h ); 
renderer_WorldView.setViewport(0,0,renderer_w, renderer_h); 

document.body.appendChild( renderer.domElement );
document.body.appendChild( renderer_WorldView.domElement );


const far = 100;
const near = 1 ;
const camera_ar = new THREE.PerspectiveCamera( 45, renderer_w / renderer_h, near, far);

const far_world = 500;
const camera_world = new THREE.PerspectiveCamera( 45, renderer_w / renderer_h, near, far_world);

camera_ar.position.set( 0, 0, 0 );
camera_ar.lookAt( 0, 0, 0 ); 
camera_ar.up.set(0,1,0);

camera_world.position.set( 5,10,-30 );
camera_world.lookAt( 0, 0, 0 ); 
camera_world.up.set(0,1,0);

const scene = new THREE.Scene();

let earth;

background1_sphere();

let cameraPerspectiveHelper = new THREE.CameraHelper( camera_ar );
// scene.add( cameraPerspectiveHelper );

const controls = new OrbitControls(camera_world,renderer_WorldView.domElement );
controls.target.set(0, 0, 9);
controls.update();

/////////////////////////////////
const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial( {color: 0x00ff00} );
const cube = new THREE.Mesh( geometry, material );
cube.position.set(0,0,-5);
scene.add( cube );
/////////////////////////////////

let hand_point = null;
let hand_line1 = null;
const l1 = [4, 3, 2, 1, 0, 17, 18, 19, 20, 19, 18, 17, 13, 14, 15, 16, 15, 14, 13, 9, 10, 11, 12, 11, 10, 9, 5, 6, 7, 8, 7, 6, 5, 0]; // Media pipe에서 표현한 손 라인

let landVector = [];
let palmCenter = new Vector3();
let palmCenterInScreen = new Vector2();
let prevPalmCenterInScreen = new Vector2();

const raycaster = new THREE.Raycaster();
let pointer = new THREE.Vector2();

let lightIntens = 5;
let light = new THREE.PointLight( 0xffffff, lightIntens , 200 );
light.position.set( 0, 0, 0 );
scene.add( light );

let thumbDirection = 0;
let IndexFingerDirection = 0;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      const num_hand_points = landmarks.length;
      if(hand_point == null){
        let hand_point_geo = new THREE.BufferGeometry();
        const hand_vertices = [];
        for(let i = 0; i < num_hand_points; i++){
          const pos_ns = landmarks[i];
          landVector.push( new THREE.Vector3(pos_ns.x,pos_ns.y,pos_ns.z));
          const pos_ps = new THREE.Vector3(-1*(pos_ns.x - 0.5) * 2, -(pos_ns.y - 0.5) * 2, pos_ns.z+0.5);
          let pos_ws = new THREE.Vector3(pos_ps.x, pos_ps.y, pos_ps.z).unproject(camera_ar);
          hand_vertices.push(pos_ws.x, pos_ws.y, pos_ws.z);
        }
        
        const point_mat = new THREE.PointsMaterial({color:0x00FFFF, size:0.2});
        hand_point_geo.setAttribute('position', new THREE.Float32BufferAttribute(hand_vertices,3));
        hand_point = new THREE.Points(hand_point_geo, point_mat); 
        scene.add(hand_point);

        const p_c = new THREE.Vector3(0,0,0).unproject(camera_ar)
        const vec_cam2center = new THREE.Vector3().subVectors(p_c, camera_ar.position);
        const center_dist = vec_cam2center.length();

        let hand_line_vertice = [];
        for(let i = 0; i < num_hand_points; i++){
          const pos_ns = landmarks[i];
          const pos_ps = new THREE.Vector3((pos_ns.x - 0.5) * -2, -(pos_ns.y - 0.5) * 2, pos_ns.z);
          let pos_ws = new THREE.Vector3(pos_ps.x, pos_ps.y, pos_ps.z).unproject(camera_ar);
        
          pos_ws = ProjScale(pos_ws, camera_ar.position, center_dist, 100.0);
          hand_line_vertice.push(pos_ws.x, pos_ws.y, pos_ws.z);
        }

        let hand_line1_geometry = new THREE.BufferGeometry();
        let l1_vertice = [];
        for (let i of l1){
          l1_vertice.push(hand_line_vertice[i*3],hand_line_vertice[i*3+1],hand_line_vertice[i*3+2]);
        }

        hand_line1_geometry.setAttribute('position', new THREE.Float32BufferAttribute(l1_vertice,3));
        const hand_line_material = new THREE.LineBasicMaterial({color: 0x00ff00});

        hand_line1 = new THREE.Line(hand_line1_geometry, hand_line_material);
        scene.add(hand_line1);
      }
      const p_c = new THREE.Vector3(0,0,0).unproject(camera_ar);
      const vec_cam2center = new THREE.Vector3().subVectors(p_c, camera_ar.position);
      const center_dist = vec_cam2center.length();

      const num_points = landmarks.length;
      let hand_vertice = [];
      let position1 = hand_point.geometry.attributes.position.array;
      for(let i =0; i< num_points; i++){
        const pos_ns = landmarks[i];
        landVector.push( new THREE.Vector3(pos_ns.x,pos_ns.y,pos_ns.z));
        const pos_ps = new THREE.Vector3((pos_ns.x -0.5)*-2,-(pos_ns.y -0.5)*2, pos_ns.z+0.5) // 그냥 z축 좀 뒤에 둠. 김머훈
        palmCenterInScreen = new THREE.Vector2(pos_ps.x, pos_ps.y);
        let pos_ws = new THREE.Vector3(pos_ps.x,pos_ps.y,pos_ps.z).unproject(camera_ar);

        pos_ws = ProjScale(pos_ws, camera_ar.position, center_dist, 7.0 );
        hand_vertice.push(pos_ws);
        position1[3*i +0] = pos_ws.x;
        position1[3*i +1] = pos_ws.y;
        position1[3*i +2] = pos_ws.z;
      }
      palmCenter = hand_vertice[9];
      thumbDirection = hand_vertice[3].y - hand_vertice[2].y;
      IndexFingerDirection = hand_vertice[8].y - hand_vertice[7].y
      
      let position2 = hand_line1.geometry.attributes.position.array;
      const num_hand_line1_points = l1.length;
      for (let i =0; i<num_hand_line1_points; i++){
        let j = l1[i]
        position2[3*i +0] = position1[3*j +0];
        position2[3*i +1] = position1[3*j +1];
        position2[3*i +2] = position1[3*j +2];
      }
      hand_point.geometry.attributes.position.array = position1;
      hand_line1.geometry.attributes.position.array = position2;
      hand_line1.geometry.attributes.position.needsUpdate = true;
      hand_point.geometry.attributes.position.needsUpdate = true;
 
      mlfist(landVector);
      landVector = [];
    }
  }
  camera_ar.updateProjectionMatrix();
  scene.remove(cameraPerspectiveHelper);
  renderer.render(scene, camera_ar);
  renderer_WorldView.render(scene,camera_world )
  canvasCtx.restore();
}


const hands = new Hands({locateFile: (file) => {
  return `./node_modules/@mediapipe/hands/${file}`;
}});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onResults);


const camera = new Camera(cameraElement, {
  onFrame: async () => {
    await hands.send({image: cameraElement});
  },
  width: 640,
  height: 480
});
camera.start();

let fist_count = -1;
let angle = [];
let a = new THREE.Vector3();
let b = new THREE.Vector3();
let c = new THREE.Vector3();
let d = new THREE.Vector3();

let worldHandPoint = new THREE.Vector3;
let prevworldHandPoint = new THREE.Vector3;
earth.matrixAutoUpdate = false;
earth.matrixWorldNeedsUpdate = true;


function mlfist(landvector){
  
  for(let i = 0; i < 15; i++){
        a.copy(landvector[i]);
        b.copy(landvector[i+1]);
        c.copy(landvector[i+2]);
        d.copy(landvector[i+3]);

        let vec1 = b.sub(a);
        let vec2 = d.sub(c);

        angle.push(Math.acos((vec1.dot(vec2))/(vec1.length() * vec2.length()))* (180/Math.PI))
  }

  const result = knn(features, labels, tf.tensor(angle), 9);
  let pos = []
  for(let i =0; i < 9; i++){
    pos.push(result[i].arraySync()[1] )
  }
  if(getMode(pos) == 0){

    fist_count += 1;

    if(fist_count == 0){
      prevworldHandPoint = palmCenter;
      prevPalmCenterInScreen =  new THREE.Vector3(pointer.x, pointer.y, -1).unproject(camera_ar);
    }
    else{
      pointer = palmCenterInScreen;
      raycaster.setFromCamera( pointer, camera_ar );
      const intersects = raycaster.intersectObjects( scene.children );

      worldHandPoint = palmCenter;
      let tempt = worldHandPoint.clone();
      
      if ( intersects[0].object == cube ){
        cube.matrixAutoUpdate = false;
        cube.matrixWorldNeedsUpdate = true;

        let screenSpaceHandPoint = new THREE.Vector3(pointer.x, pointer.y, -1).unproject(camera_ar);
        let handMoveInWorld = screenSpaceHandPoint.sub(prevPalmCenterInScreen);
        handMoveInWorld.multiplyScalar(5);
        let m = new THREE.Matrix4();
        m.makeTranslation(( handMoveInWorld.x),( handMoveInWorld.y),( handMoveInWorld.z));

        let cube_mat_prev = cube.matrix.clone();
        cube_mat_prev.premultiply(m);
        cube.matrix.copy(cube_mat_prev);
        cube.updateMatrixWorld();
      }
      else{
        let v1 = worldHandPoint.clone();
        let v2 = prevworldHandPoint.clone();
        let d1 = worldHandPoint.length();
        let d2 = prevworldHandPoint.length();
        let dotProduct = v1.dot(v2);
        let theta = Math.acos(dotProduct/(d1*d2));
        prevworldHandPoint.cross(worldHandPoint);
        let a = new THREE.Matrix4().makeRotationFromQuaternion(
          new THREE.Quaternion().setFromAxisAngle(prevworldHandPoint.normalize(), Math.abs(theta))
          );
        let earth_prev = earth.matrix.clone();
        earth_prev.premultiply(a);
        earth.matrix.copy(earth_prev);
        earth.updateMatrixWorld()
      }
      prevworldHandPoint = tempt;
    }
  }
  else if(getMode(pos) == 2 && thumbDirection < 0){
    light.intensity = lightIntens;
    lightIntens -= 0.1;
  }
  else if(getMode(pos) == 3 && IndexFingerDirection > 0){
    light.intensity = lightIntens;
    lightIntens += 0.1;
  }
  else{
    fist_count = -1;
  }
  prevPalmCenterInScreen = new THREE.Vector3(palmCenterInScreen.x, palmCenterInScreen.y, -1).unproject(camera_ar);
   
  angle = [];
}


function getMode(array){
  // 1. 출연 빈도 구하기
  const counts = array.reduce((pv, cv)=>{
      pv[cv] = (pv[cv] || 0) + 1;
      return pv;
  }, {});

  // 2. 최빈값 구하기
  const keys = Object.keys(counts);
  let mode = keys[0];
  keys.forEach((val, idx)=>{
      if(counts[val] > counts[mode]){
          mode = val;
      }
  });

  return mode;
}

window.addEventListener("keydown", (e) => console.log(angle)); 