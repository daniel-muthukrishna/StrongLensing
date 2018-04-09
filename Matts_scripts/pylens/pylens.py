class LensModel:
    """
    IN PROGRESS
    The purpose of this class is to allow lens modelling with e.g., caching to
        update parts of the model independently.
    """
    def __init__(self,img,sig,gals,lenses,srcs,psf=None):
        self.img = img
        self.sig = sig
        self.gals = gals
        self.lenses = lenses
        self.srcs = srcs
        self.psf = psf

        self.rhs = (img/sig).flatten()
        self.fsig = sig.flatten()
        self.fit

        self.mask = None

        self.logp = None
        self.model = None
        self.amps = None
        self.components = None

    def addMask(self,mask):
        self.rhs = (img/sig)[mask].flatten()
        self.fsig = sig[mask].flatten()

    def fit(x,y,fast=False):
        mask = self.mask
        gals = self.gals
        srcs = self.srcs
        lenses = self.lenses
        rhs = self.rhs
        sig = self.fsig

        if mask is not None:
            xin = x[mask].copy()
            yin = y[mask].copy()
        else:
            xin = x.copy()
            yin = y.copy()

        model = numpy.empty((len(gals)+len(srcs),rhs.size))
        n = 0
        for gal in gals:
            gal.setPars()
            gal.amp = 1.
            tmp = gal.pixeval(xin,yin,1./self.oversample,csub=self.csub)
            if numpy.isnan(tmp):
                self.model = None
                self.amps = None
                self.components = None
                self.logp = -1e300
                return -1e300
            if psf is not None and gal.convolve is not None:
                if mask is None:
                    model[n] = convolve.convolve(tmp,psf,False)[0]
                else:
                    model[n] = 1 
    
        gals = self.gals


def getDeflections(massmodels,points, d=1):
    if type(points)==type([]) or type(points)==type(()):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    if type(massmodels)!=type([]):
        massmodels = [massmodels]
    x0 = x.copy()
    y0 = y.copy()
    for massmodel in massmodels:
        xmap,ymap = massmodel.deflections(x,y, d)
        y0 -= ymap
        x0 -= xmap
    return x0.reshape(x.shape),y0.reshape(y.shape)



def lens_images(massmodels,sources,points,factor=1,getPix=False):
    if type(points)==type([]):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    if type(massmodels)!=type([]):
        massmodels = [massmodels]
#    x0 = x.flatten()
#    y0 = y.flatten()
    x0 = x.copy()
    y0 = y.copy()
    for massmodel in massmodels:
        xmap,ymap = massmodel.deflections(x,y)
        y0 -= ymap#.reshape(y.shape)#/scale
        x0 -= xmap#.reshape(x.shape)#/scale
    x0,y0 = x0.reshape(x.shape),y0.reshape(y.shape)
    if getPix==True:
        return x0,y0
    if type(sources)!=type([]):
        sources = [sources]
    out = x*0.
    for src in sources:
        out += src.pixeval(x0,y0,factor,csub=11)
    return out


def dblPlane(scales,massmodels,sources,points,factor):
    if type(points)==type([]):
        x1,y1 = points[0].copy(),points[1].copy()
    else:
        y1,x1 = points[0].copy(),points[1].copy()
    out = x1*0.
    ax_1 = x1*0.
    ay_1 = x1*0.
    for l in massmodels[0]:
        xmap,ymap = l.deflections(x1,y1)
        ax_1 += xmap.reshape(ax_1.shape)
        ay_1 += ymap.reshape(ay_1.shape)
    for s in sources[0]:
        out += s.pixeval(x1-ax_1,y1-ay_1,factor,csub=11)
    x2 = x1-scales[0,0]*ax_1
    y2 = y1-scales[0,0]*ay_1
    ax_2 = x2*0.
    ay_2 = y2*0.
    for l in massmodels[1]:
        xmap,ymap = l.deflections(x2,y2)
        ax_2 += xmap.reshape(ax_2.shape)
        ay_2 += ymap.reshape(ay_2.shape)
    for s in sources[1]:
        out += s.pixeval(x1-ax_1-ax_2,y1-ay_1-ay_2,factor,csub=11)
    return out


def multiplePlanes(scales,massmodels,points):
    from numpy import zeros,eye,triu_indices
    if type(points)==type([]):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    shp = x.shape
    x = x.ravel()
    y = y.ravel()

    nplanes = len(massmodels)
    tmp = scales.copy()
    scales = zeros((nplanes,nplanes))
    scales[triu_indices(nplanes)] = tmp
    ax_i = zeros((x.size,nplanes))
    ay_i = ax_i.copy()
    x_i = x.flatten()
    y_i = y.flatten()
    out = []
    for p in range(nplanes):
        for massmodel in massmodels[p]:
            xmap,ymap = massmodel.deflections(x_i,y_i)
            ax_i[:,p] += xmap#.reshape(x.shape)
            ay_i[:,p] += ymap#.reshape(y.shape)
        x_i = x-(ax_i[:,:p+1]*scales[:p+1,p])[:,:p+1].sum(1)
        y_i = y-(ay_i[:,:p+1]*scales[:p+1,p])[:,:p+1].sum(1)
#        out.append([x_i.reshape(shp),y_i.reshape(shp)])
    return x_i.reshape(shp),y_i.reshape(shp)


def omultiplePlanes(scales,massmodels,points):
    from numpy import zeros,eye,triu_indices
    if type(points)==type([]):
        x,y = points[0].copy(),points[1].copy()
    else:
        y,x = points[0].copy(),points[1].copy()
    out = x*0.
    nplanes = len(massmodels)
    tmp = scales.copy()
    scales = eye(nplanes)
    scales[triu_indices(nplanes,1)] = tmp
    ax = zeros((x.shape[0],x.shape[1],nplanes))
    ay = ax.copy()
    x0 = x.copy()
    y0 = y.copy()
    out = []
    for p in range(nplanes):
        for massmodel in massmodels[p]:
            xmap,ymap = massmodel.deflections(x0,y0)
            ax[:,:,p] += xmap.reshape(x.shape)
            ay[:,:,p] += ymap.reshape(y.shape)
        x0 = x-(ax[:,:,:p+1]*scales[:p+1,p]).sum(2)
        y0 = y-(ay[:,:,:p+1]*scales[:p+1,p]).sum(2)
        out.append([x0,y0])
    return out



def getImgPos(x0,y0,b,sx,sy,lenses,scales=None,N=4,tol=1e-5,DEBUG=None):
    import numpy,pylab
    from scipy import ndimage

    def getDefs(xt,yt):
        if scales is not None:
            xs0,ys0 = multiplePlanes(scales,lenses,[xt,yt])
        else:
            xs0,ys0 = getDeflections(lenses,[xt,yt])
        return xs0,ys0


    # r spacing
    deltaR = 0.1
    # theta spacing
    deltaT = 3.


    r = numpy.array([0.05,0.15,0.25,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.925,0.95,0.975,1.,1.025,1.05,1.075,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.95,2.05,2.15])
    r = numpy.arange(0.05,2.15+deltaR/2,deltaR)
    t = numpy.arange(0.,360.-deltaT/2.,deltaT)*numpy.pi/180
    theta,r = numpy.meshgrid(t,r)

    tol *= b


    xt = b*r*numpy.cos(theta)+x0
    yt = b*r*numpy.sin(theta)+y0

    xs,ys = getDefs(xt,yt)
    d = ((xs-sx)**2+(ys-sy)**2)/b**2
    logd = numpy.log10(d)

    delD = ndimage.laplace(logd,mode='wrap')
    f1 = ndimage.maximum_filter(delD,3,mode='wrap')
    f2 = ndimage.minimum_filter(logd,3,mode='wrap')
    peaks = (f1>0.5)&(f2==logd)&(logd<-2)


    # Hack to label through the wrapped boundary
    def labelWrap(arr,n=2):
        tmp,nreg = ndimage.label(arr,structure=numpy.ones((3,3)))
        left = tmp[:,:n].copy()
        right = tmp[:,-n:][:,::-1].copy()
        overlap = left*right
        diffs = numpy.unique(overlap[overlap!=0])
        for d in diffs:
            l = left[overlap==d][0]
            r = right[overlap==d][0]
            if l==r:
                continue
            tmp[tmp==r] = l
            nreg -= 1
        vals = numpy.sort(numpy.unique(tmp[tmp!=0]))
        for i in range(nreg):
            if vals[i]==i+1:
                continue
            tmp[tmp==vals[i]] = i+1
        return tmp,nreg


    # Locate minimal _regions_
    regions = peaks*0
    c = 1
    iy,ix = numpy.where(peaks)
    mymap = labelWrap(f1>0.5)[0]
    for i in range(ix.size):
        y = iy[i]
        x = ix[i]
#        dpeak = logd[y,x]
#        mymap = labelWrap(delD>0.5)[0]
        tmp = mymap==mymap[y,x]
        regions[tmp] += c
        c += 1
    labels,nreg = labelWrap(regions)


    if DEBUG:
        d10 = numpy.log10(d)
        pylab.imshow(d10,origin='lower',interpolation='nearest')
        pylab.figure()
        pylab.imshow(ndimage.laplace(d10),origin='lower',interpolation='nearest')
        pylab.figure()
        pylab.imshow(regions,origin='lower')
        pylab.figure()
        pylab.imshow(labels,origin='lower')
        pylab.show()

    # Find the minimum in each region
    ix,iy = [],[]
    for i in range(nreg):
        ym,xm = numpy.where(d==d[labels==i+1].min())
        ix.append(xm[0])
        iy.append(ym[0])



    def findPeaks(i,labels,rorig,torig,depth=0):
        if depth>10:
            return ()
        cond = labels==i
        if depth>0:
            cond = ndimage.maximum_filter(cond,3)

        t0 = torig[cond].copy()
        r0 = rorig[cond].copy()
        t1,t2 = t0.min(),t0.max()
        r1,r2 = r0.min(),r0.max()

        ndim = 4*N+1+depth*4*N

        if depth==0:
            dt = numpy.linspace(t1,t2,ndim+4*N)
            dr = numpy.linspace(r1,r2,ndim+4*N)
        else:
            dt = numpy.linspace(t1,t2,ndim)
            dr = numpy.linspace(r1,r2,ndim)

        if t2-t1>numpy.pi:
            dt = numpy.linspace(t1,t2,dt.size*2)

        # Check if we've wrapped
        if t1==theta.min() and t2==theta.max():
            tmp = cond.sum(0)>0
            if tmp.sum()!=tmp.size:
                t0 = torig[0,tmp==0]
                t1 = t0[-1]-2*numpy.pi
                t2 = t0[0]
                if depth==0:
                    dt = numpy.linspace(t1,t2,ndim+4*N)
                else:
                    dt = numpy.linspace(t1,t2,ndim)
            else:
                t1 = theta.min()
                t2 = theta.max()
                dt = numpy.linspace(t1,t2,360)

        t0,r0 = numpy.meshgrid(dt,dr)
        xt = b*r0*numpy.cos(t0)+x0
        yt = b*r0*numpy.sin(t0)+y0

        xs0,ys0 = getDefs(xt,yt)

        d0 = numpy.log10(((xs0-sx)**2+(ys0-sy)**2)/b**2)

        f1 = ndimage.minimum_filter(d0,footprint=[[0,1,0],[1,1,1],[0,1,0]])==d0
        img = ndimage.minimum_filter(f1*d0,3)
        peaks = (img==d0)&(d0<-2-depth)

#        peaks = (ndimage.minimum_filter(d0,3)==d0)&(d0<-2-depth)

        if peaks.sum()==1:
            iy,ix = numpy.where(peaks)
            rgrid = (r0-r0.mean())/N
            tgrid = (t0-t0.mean())/N

            xs0,ys0 = xs0[iy,ix],ys0[iy,ix]
            r0,t0 = r0[iy,ix],t0[iy,ix]

            dt = (t2-t1)/2
            dr = (r2-r1)/2
            dt = numpy.linspace(-dt,dt,ndim)
            dr = numpy.linspace(-dr,dr,ndim)
            tgrid,rgrid = numpy.meshgrid(dt,dr)
#            rgrid = (r0-r0.mean())/N
#            tgrid = (t0-t0.mean())/N

            count = 0
            while count==0 or abs(xs0-sx)>tol or abs(ys0-sy)>tol:
                rgrid /= N
                tgrid /= N

                rt = r0+rgrid
                tt = t0+tgrid
                xt = b*rt*numpy.cos(tt)+x0
                yt = b*rt*numpy.sin(tt)+y0

                xst,yst = getDefs(xt,yt)

                dd = ((xst-sx)**2+(yst-sy)**2)/b**2

                iy,ix = numpy.unravel_index(dd.argmin(),dd.shape)

                dmin = dd[iy,ix]
                xs0,ys0 = xst[iy,ix],yst[iy,ix]
                r0,t0 = rt[iy,ix],tt[iy,ix]

                count += 1
                if count>=10:
                    break

            if count<10:
                # Calculate magnification
                eps = 1e-7
                amp = 1
                
                xf,yf = xt[iy,ix],yt[iy,ix]
                dx = numpy.array([0.,0.,eps,-eps])
                dy = numpy.array([eps,-eps,0.,0.])
                ox,oy = getDefs(xf+dx,yf+dy)
                dxx = (ox[2]-ox[3])/(2*eps)
                dxy = (ox[0]-ox[1])/(2*eps)
                dyx = (oy[2]-oy[3])/(2*eps)
                dyy = (oy[0]-oy[1])/(2*eps)
                amp = 1./(dxx*dyy-dxy*dyx)
                return xt[iy,ix],yt[iy,ix],amp,count


            if DEBUG:
                pylab.figure()
                pylab.imshow(dd**0.5,origin='lower',interpolation='nearest',extent=[tgrid.min(),tgrid.max(),rgrid.min(),rgrid.max()])
                pylab.figure()
                pylab.imshow(d0,origin='lower',interpolation='nearest')
                pylab.show()

            return ()


        # Check for rare case of needing to wrap the labelling
        if dt.size==360:
            labelFunc = labelWrap
        else:
            labelFunc = lambda f: ndimage.label(f,structure=numpy.ones((3,3)))


        regions = peaks*0
        c = 1
        for val in numpy.sort(d0[peaks])[::-1]:
            tmp = labelFunc(d0<val+1)[0]
            tmp = tmp==tmp[d0==val]
            regions[tmp] += c
            c += 1
        labels,nreg = labelFunc(regions)


        if DEBUG:
            if depth>4:
                print xt.mean(),yt.mean()
            pylab.figure()
            pylab.imshow(d0,origin='lower',interpolation='nearest')
            pylab.figure()
            pylab.imshow(labels,origin='lower',interpolation='nearest')
            pylab.figure()
            pylab.imshow(img,origin='lower',interpolation='nearest')
            pylab.figure()
            pylab.imshow(regions,origin='lower',interpolation='nearest')
            pylab.show()

        result = []
        for j in range(nreg):
            result.append(findPeaks(j+1,labels,r0,t0,depth+1))

        return result

    minima = []
    for i in range(nreg):
        minima.append(findPeaks(i+1,labels,r,theta))


    def flatten(container):
        for i in container:
            if isinstance(i, tuple):
                if len(i)>0:
                    yield i
            else:
                for j in flatten(i):
                    yield j

    return list(flatten(minima))



