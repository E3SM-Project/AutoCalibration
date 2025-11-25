# an alternate version of plotrix::taylor.diagram
# allows one to specify weights in computation, and 

taylor.diagram2 <- function (ref, model, wts = rep(1, length(ref)), add = FALSE, col = "red", pch = 19, pos.cor = TRUE, 
          xlab = "Standard deviation", ylab = "", main = "Taylor Diagram", 
          show.gamma = TRUE, ngamma = 3, gamma.col = 8, sd.arcs = 0, 
          ref.sd = FALSE, sd.method = "sample", grad.corr.lines = c(0.2, 
                                                                    0.4, 0.6, 0.8, 0.9), pcex = 1, cex.axis = 1, normalize = FALSE, 
          mar = c(4, 3, 4, 3), ...) {
  grad.corr.full <- c(0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 
                      1)
  complete_obs <- !is.na(ref) & !is.na(model)
  ref <- ref[complete_obs]
  model <- model[complete_obs]
  wts <- wts[complete_obs]
  R <- cor(ref, model, use = "pairwise")
  cor(ref, model, use = "pairwise", method = 'pearson')
  #wts <- rep(1, length(ref))
  sd_ref <- sqrt(1/(length(ref) - 1) * sum(wts * (ref - mean(ref))^2))
  sd_model <- sqrt(1/(length(model) - 1) * sum(wts * (model - mean(model))^2))
  R <- sum((ref - mean(ref)) * (model- mean(model)) * wts)/sd_ref/sd_model/(length(ref)-1)
  #sum((ref - mean(ref)) * (model- mean(model)) * wts)/sd(ref)/sd(model)/(length(ref)-1)
  #cov(ref, model)/sd(ref)/sd(model)
  if (is.list(ref)) 
    ref <- unlist(ref)
  if (is.list(model)) 
    ref <- unlist(model)
  SD <- function(x, subn, wts) {
    meanx <- mean(x, na.rm = TRUE)
    devx <- x - meanx
    ssd <- sqrt(sum(devx * devx * wts, na.rm = TRUE)/(length(x[!is.na(x)]) - 
                                                  subn))
    return(ssd)
  }
  subn <- sd.method != "sample"
  sd.r <- SD(ref, subn, wts)
  sd.f <- SD(model, subn, wts)
  if (normalize) {
    sd.f <- sd.f/sd.r
    sd.r <- 1
  }
  maxsd <- 1.5 * max(sd.f, sd.r)
  oldpar <- par("mar", "xpd", "xaxs", "yaxs")
  if (!add) {
    par(mar = mar)
    if (pos.cor) {
      if (nchar(ylab) == 0) 
        ylab = "Standard deviation"
      plot(0, xlim = c(0, maxsd * 1.1), ylim = c(0, maxsd * 
                                                   1.1), xaxs = "i", yaxs = "i", axes = FALSE, main = main, 
           xlab = "", ylab = ylab, type = "n", cex = cex.axis, 
           ...)
      mtext(xlab, side = 1, line = 2.3)
      if (grad.corr.lines[1]) {
        for (gcl in grad.corr.lines) lines(c(0, maxsd * 
                                               gcl), c(0, maxsd * sqrt(1 - gcl^2)), lty = 3)
      }
      segments(c(0, 0), c(0, 0), c(0, maxsd), c(maxsd, 
                                                0))
      axis.ticks <- pretty(c(0, maxsd))
      axis.ticks <- axis.ticks[axis.ticks <= maxsd]
      axis(1, at = axis.ticks, cex.axis = cex.axis)
      axis(2, at = axis.ticks, cex.axis = cex.axis)
      if (sd.arcs[1]) {
        if (length(sd.arcs) == 1) 
          sd.arcs <- axis.ticks
        for (sdarc in sd.arcs) {
          xcurve <- cos(seq(0, pi/2, by = 0.03)) * sdarc
          ycurve <- sin(seq(0, pi/2, by = 0.03)) * sdarc
          lines(xcurve, ycurve, col = "blue", lty = 3)
        }
      }
      if (show.gamma[1]) {
        if (length(show.gamma) > 1) 
          gamma <- show.gamma
        else gamma <- pretty(c(0, maxsd), n = ngamma)[-1]
        if (gamma[length(gamma)] > maxsd) 
          gamma <- gamma[-length(gamma)]
        labelpos <- seq(45, 70, length.out = length(gamma))
        for (gindex in 1:length(gamma)) {
          xcurve <- cos(seq(0, pi, by = 0.03)) * gamma[gindex] + 
            sd.r
          endcurve <- which(xcurve < 0)
          endcurve <- ifelse(length(endcurve), min(endcurve) - 
                               1, 105)
          ycurve <- sin(seq(0, pi, by = 0.03)) * gamma[gindex]
          maxcurve <- xcurve * xcurve + ycurve * ycurve
          startcurve <- which(maxcurve > maxsd * maxsd)
          startcurve <- ifelse(length(startcurve), max(startcurve) + 
                                 1, 0)
          lines(xcurve[startcurve:endcurve], ycurve[startcurve:endcurve], 
                col = gamma.col)
          if (xcurve[labelpos[gindex]] > 0) 
            boxed.labels(xcurve[labelpos[gindex]], ycurve[labelpos[gindex]], 
                         gamma[gindex], border = FALSE)
        }
      }
      xcurve <- cos(seq(0, pi/2, by = 0.01)) * maxsd
      ycurve <- sin(seq(0, pi/2, by = 0.01)) * maxsd
      lines(xcurve, ycurve)
      bigtickangles <- acos(seq(0.1, 0.9, by = 0.1))
      medtickangles <- acos(seq(0.05, 0.95, by = 0.1))
      smltickangles <- acos(seq(0.91, 0.99, by = 0.01))
      segments(cos(bigtickangles) * maxsd, sin(bigtickangles) * 
                 maxsd, cos(bigtickangles) * 0.97 * maxsd, sin(bigtickangles) * 
                 0.97 * maxsd)
      par(xpd = TRUE)
      if (ref.sd) {
        xcurve <- cos(seq(0, pi/2, by = 0.01)) * sd.r
        ycurve <- sin(seq(0, pi/2, by = 0.01)) * sd.r
        lines(xcurve, ycurve)
      }
      points(sd.r, 0, cex = pcex)
      text(cos(c(bigtickangles, acos(c(0.95, 0.99)))) * 
             1.05 * maxsd, sin(c(bigtickangles, acos(c(0.95, 
                                                       0.99)))) * 1.05 * maxsd, c(seq(0.1, 0.9, by = 0.1), 
                                                                                  0.95, 0.99), cex = cex.axis)
      text(maxsd * 0.8, maxsd * 0.8, "Correlation", srt = 315, 
           cex = cex.axis)
      segments(cos(medtickangles) * maxsd, sin(medtickangles) * 
                 maxsd, cos(medtickangles) * 0.98 * maxsd, sin(medtickangles) * 
                 0.98 * maxsd)
      segments(cos(smltickangles) * maxsd, sin(smltickangles) * 
                 maxsd, cos(smltickangles) * 0.99 * maxsd, sin(smltickangles) * 
                 0.99 * maxsd)
    } 
  }
  if (show.gamma[1]) {
    if (length(show.gamma) > 1) 
      gamma <- show.gamma
    else gamma <- pretty(c(0, maxsd), n = ngamma)[-1]
    if (gamma[length(gamma)] > maxsd) 
      gamma <- gamma[-length(gamma)]
    labelpos <- seq(45, 70, length.out = length(gamma))
    for (gindex in 1:length(gamma)) {
      xcurve <- cos(seq(0, pi, by = 0.03)) * gamma[gindex] + 
        sd.r
      endcurve <- which(xcurve < 0)
      endcurve <- ifelse(length(endcurve), min(endcurve) - 
                           1, 105)
      ycurve <- sin(seq(0, pi, by = 0.03)) * gamma[gindex]
      maxcurve <- xcurve * xcurve + ycurve * ycurve
      startcurve <- which(maxcurve > maxsd * maxsd)
      startcurve <- ifelse(length(startcurve), max(startcurve) + 
                             1, 0)
      lines(xcurve[startcurve:endcurve], ycurve[startcurve:endcurve], 
            col = gamma.col)
      if (xcurve[labelpos[gindex]] > 0) 
        boxed.labels(xcurve[labelpos[gindex]], ycurve[labelpos[gindex]], 
                     gamma[gindex], border = FALSE)
    }
  }
  points(sd.f * R, sd.f * sin(acos(R)), pch = pch, col = col, 
         cex = pcex)
  invisible(oldpar)
}
