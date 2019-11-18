function view_recon(recon,dim)
    recon = abs(recon)/max(max(max(abs(recon))));
    i=1;
    if(dim==3)
        while(1)
            imagesc(squeeze(recon(:,:,i))',[0,1]);
            colormap('gray');
            waitforbuttonpress;
            i=i+1;
            if(i==size(recon,3))
                i=1;
            end
        end
    elseif(dim==2)
        while(1)
            imagesc(squeeze(recon(:,i,:))',[0,1]);
            colormap('gray');
            waitforbuttonpress;
            i=i+1;
            if(i==size(recon,3))
                i=1;
            end
        end
    elseif(dim==1)
        while(1)
            imagesc(squeeze(recon(i,:,:))',[0,1]);
            colormap('gray');
            waitforbuttonpress;
            i=i+1;
            if(i==size(recon,3))
                i=1;
            end
        end
    end
end